// The GPU signals and waits for the host inside MLX's command stream through an MTLSharedEvent, never an MLX eval.

#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <functional>
#include <list>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>

#include "mlx/backend/metal/device.h"
#include "mlx/mlx.h"
#include "mlx/primitives.h"

namespace nb = nanobind;
namespace mx = mlx::core;

// One monotonic event shared by the GPU and the host: each side signals increasing values.
struct Channel {
  NS::SharedPtr<MTL::SharedEvent> event;
  Channel() : event(NS::TransferPtr(mx::metal::device(mx::Device::gpu).mtl_device()->newSharedEvent())) {}
  uint64_t value() const { return event->signaledValue(); }
  void signal(uint64_t v) { event->setSignaledValue(v); }
  bool wait(uint64_t v, uint64_t timeout_ms) {
    nb::gil_scoped_release release;
    return event->waitUntilSignaledValue(v, timeout_ms);
  }
};

// Passes its input through; on the GPU it first ends the compute encoder, then waits for or signals the channel.
class Mark : public mx::UnaryPrimitive {
 public:
  Mark(mx::Stream s, std::shared_ptr<Channel> channel, uint64_t value, bool wait)
      : mx::UnaryPrimitive(s), channel_(std::move(channel)), value_(value), wait_(wait) {}
  void eval_cpu(const std::vector<mx::array>&, mx::array&) override {
    throw std::runtime_error("hostsync marks run on the GPU stream only");
  }
  void eval_gpu(const std::vector<mx::array>& inputs, mx::array& out) override {
    out.copy_shared_buffer(inputs[0]);
    auto& enc = mx::metal::get_command_encoder(stream());
    enc.end_encoding();
    if (wait_) {
      enc.get_command_buffer()->encodeWait(channel_->event.get(), value_);
    } else {
      enc.get_command_buffer()->encodeSignalEvent(channel_->event.get(), value_);
    }
  }
  const char* name() const override { return wait_ ? "HostWait" : "HostSignal"; }
  bool is_equivalent(const mx::Primitive& other) const override {
    auto* o = dynamic_cast<const Mark*>(&other);
    return o && o->channel_ == channel_ && o->value_ == value_ && o->wait_ == wait_;
  }

 private:
  std::shared_ptr<Channel> channel_;
  uint64_t value_;
  bool wait_;
};

mx::array mark(const mx::array& x, std::shared_ptr<Channel> channel, uint64_t value, bool wait) {
  auto s = mx::default_stream(mx::Device::gpu);
  return mx::array(x.shape(), x.dtype(), std::make_shared<Mark>(s, std::move(channel), value, wait), {x});
}

// Reads nbytes of fd at file_offset into an evaluated array's buffer at byte_offset (the caller owns the timing).
void pread_into(mx::array& a, size_t byte_offset, int fd, int64_t file_offset, size_t nbytes) {
  if (!a.is_available() || byte_offset + nbytes > a.nbytes()) {
    throw std::out_of_range("pread_into: the array is not evaluated or the range passes its end");
  }
  auto* dst = a.data<uint8_t>() + byte_offset;
  nb::gil_scoped_release release;
  size_t done = 0;
  while (done < nbytes) {
    ssize_t got = pread(fd, dst + done, nbytes - done, file_offset + static_cast<int64_t>(done));
    if (got < 0 && errno == EINTR) continue;
    if (got <= 0) throw std::runtime_error(std::string("pread_into: short read: ") + std::strerror(errno));
    done += static_cast<size_t>(got);
  }
}

// Copies bytes into an evaluated array's buffer at byte_offset (host writes the GPU reads after a later wait).
void write_into(mx::array& a, size_t byte_offset, nb::bytes data) {
  if (!a.is_available() || byte_offset + data.size() > a.nbytes()) {
    throw std::out_of_range("write_into: the array is not evaluated or the range passes its end");
  }
  std::memcpy(a.data<uint8_t>() + byte_offset, data.c_str(), data.size());
}

// Reads nbytes of an evaluated array's buffer at byte_offset without waiting on MLX (after a GPU signal).
nb::bytes peek(mx::array& a, size_t byte_offset, size_t nbytes) {
  if (!a.is_available() || byte_offset + nbytes > a.nbytes()) {
    throw std::out_of_range("peek: the array is not evaluated or the range passes its end");
  }
  return nb::bytes(reinterpret_cast<const char*>(a.data<uint8_t>() + byte_offset), nbytes);
}

// An error the pool server records for Python, with the Python exception type it maps to.
struct PoolError : std::runtime_error {
  std::string kind;
  PoolError(std::string k, const std::string& msg) : std::runtime_error(msg), kind(std::move(k)) {}
};

// Reads the whole range or throws: a short read is an error, EINTR is retried.
static void pread_full(int fd, uint8_t* dst, size_t nbytes, int64_t offset) {
  size_t done = 0;
  while (done < nbytes) {
    ssize_t got = pread(fd, dst + done, nbytes - done, offset + static_cast<int64_t>(done));
    if (got < 0 && errno == EINTR) continue;
    if (got <= 0) {
      throw PoolError("RuntimeError", std::string("pread_into: short read: ") + (got < 0 ? std::strerror(errno) : "end of file"));
    }
    done += static_cast<size_t>(got);
  }
}

// A fixed set of reader threads; run() hands them one batch of tasks and returns once every task ended.
class Readers {
 public:
  explicit Readers(int n) {
    for (int i = 0; i < n; ++i) threads_.emplace_back([this, i] { loop(i); });
  }
  ~Readers() {
    {
      std::lock_guard<std::mutex> lk(mu_);
      stop_ = true;
    }
    cv_.notify_all();
    for (auto& t : threads_) t.join();
  }
  int size() const { return static_cast<int>(threads_.size()); }

  // fn(task, worker) for task in [0, count); the first error is rethrown after all tasks ended.
  void run(size_t count, const std::function<void(size_t, int)>& fn) {
    if (count == 0) return;
    auto batch = std::make_shared<Batch>();
    batch->count = count;
    batch->fn = &fn;
    {
      std::lock_guard<std::mutex> lk(mu_);
      current_ = batch;
      ++generation_;
    }
    cv_.notify_all();
    std::unique_lock<std::mutex> lk(batch->mu);
    batch->cv.wait(lk, [&] { return batch->finished == batch->count; });
    if (batch->error) std::rethrow_exception(batch->error);
  }

 private:
  struct Batch {
    size_t count = 0;
    const std::function<void(size_t, int)>* fn = nullptr;
    std::atomic<size_t> next{0};
    std::mutex mu;
    std::condition_variable cv;
    size_t finished = 0;
    std::exception_ptr error;
  };

  void loop(int id) {
    uint64_t seen = 0;
    while (true) {
      std::shared_ptr<Batch> batch;
      {
        std::unique_lock<std::mutex> lk(mu_);
        cv_.wait(lk, [&] { return stop_ || generation_ != seen; });
        if (stop_) return;
        seen = generation_;
        batch = current_;
      }
      size_t i;
      while ((i = batch->next.fetch_add(1)) < batch->count) {   // a late worker on an old batch takes nothing
        std::exception_ptr error;
        try {
          (*batch->fn)(i, id);
        } catch (...) {
          error = std::current_exception();
        }
        std::lock_guard<std::mutex> lk(batch->mu);
        if (error && !batch->error) batch->error = error;
        if (++batch->finished == batch->count) batch->cv.notify_all();
      }
    }
  }

  std::vector<std::thread> threads_;
  std::mutex mu_;
  std::condition_variable cv_;
  std::shared_ptr<Batch> current_;
  uint64_t generation_ = 0;
  bool stop_ = false;
};

// The expert pool's host side in C++: the Python Streamer's serve loop, LRU, layer window and reads, with no
// Python per call. Slots [0, E) hold one layer's window, [E, S) the decode LRU (same order and eviction as Python).
class PoolServer {
 public:
  PoolServer(std::shared_ptr<Channel> channel, mx::array box, mx::array slot_of, std::vector<mx::array> dsts,
             std::vector<size_t> part_nbytes, nb::bytes fds, nb::bytes offsets, int layers, int experts, int top_k,
             int slots, int workers, uint64_t timeout_ms)
      : channel_(std::move(channel)), box_(std::move(box)), slot_of_(std::move(slot_of)), dsts_(std::move(dsts)),
        part_nbytes_(std::move(part_nbytes)), layers_(layers), experts_(experts), top_k_(top_k), slots_(slots),
        timeout_ms_(timeout_ms) {
    size_t parts = dsts_.size();
    if (parts == 0 || part_nbytes_.size() != parts || layers <= 0 || experts <= 0 || slots <= experts || workers <= 0) {
      throw std::invalid_argument("PoolServer: bad geometry");
    }
    size_t entries = static_cast<size_t>(layers) * experts * parts;
    if (fds.size() != entries * sizeof(int32_t) || offsets.size() != entries * sizeof(int64_t)) {
      throw std::invalid_argument("PoolServer: the source table does not match layers x experts x parts");
    }
    fds_.resize(entries);
    offsets_.resize(entries);
    std::memcpy(fds_.data(), fds.c_str(), fds.size());
    std::memcpy(offsets_.data(), offsets.c_str(), offsets.size());
    for (size_t p = 0; p < parts; ++p) {
      if (!dsts_[p].is_available() || dsts_[p].nbytes() < static_cast<size_t>(slots) * part_nbytes_[p]) {
        throw std::invalid_argument("PoolServer: a pool array is not evaluated or smaller than slots x part bytes");
      }
      dst_ptrs_.push_back(dsts_[p].data<uint8_t>());
      expert_nbytes_ += part_nbytes_[p];
    }
    if (!box_.is_available() || !slot_of_.is_available() ||
        slot_of_.nbytes() < static_cast<size_t>(layers) * experts * sizeof(int32_t) ||
        box_.nbytes() < static_cast<size_t>(experts) * sizeof(uint32_t)) {
      throw std::invalid_argument("PoolServer: the box or slot table is not evaluated or too small");
    }
    window_.assign(experts, -1);
    for (int s = slots - 1; s >= experts; --s) free_.push_back(s);
    readers_ = std::make_unique<Readers>(workers);
    staging_.resize(workers);
    thread_ = std::thread([this] { serve(); });
  }

  ~PoolServer() { close(); }

  // Reads each missing expert with one pread of the pack's record, then copies its parts into their slots.
  void set_pack(int fd, nb::bytes offsets, size_t read_nbytes, std::vector<size_t> part_offsets, size_t alignment) {
    std::lock_guard<std::mutex> lk(state_mu_);
    size_t records = static_cast<size_t>(layers_) * experts_;
    if (offsets.size() != records * sizeof(int64_t) || part_offsets.size() != dsts_.size() || alignment < 4096 ||
        (alignment & (alignment - 1)) != 0 || read_nbytes % alignment != 0) {
      throw std::invalid_argument("set_pack: the pack index does not match the pool");
    }
    for (size_t p = 0; p < part_offsets.size(); ++p) {
      if (part_offsets[p] + part_nbytes_[p] > read_nbytes) throw std::invalid_argument("set_pack: a part passes its record");
    }
    pack_offsets_.resize(records);
    std::memcpy(pack_offsets_.data(), offsets.c_str(), offsets.size());
    pack_fd_ = fd;
    pack_read_ = read_nbytes;
    pack_parts_ = std::move(part_offsets);
    pack_align_ = alignment;
    for (auto& buf : staging_) {
      std::free(buf);
      buf = nullptr;
    }
  }

  // One MoE call: kind 0 = decode rows (rows x top_k ids in the box), 1 = window (E presence flags).
  void submit(int kind, int layer, int rows, uint64_t signal, uint64_t ready) {
    {
      std::lock_guard<std::mutex> lk(jobs_mu_);
      if (closed_) throw std::runtime_error("PoolServer: closed");
      jobs_.push_back(Job{kind, layer, rows, signal, ready, false});
    }
    jobs_cv_.notify_all();
  }

  std::optional<std::pair<std::string, std::string>> error() {
    std::lock_guard<std::mutex> lk(error_mu_);
    return error_;
  }

  std::tuple<uint64_t, uint64_t, uint64_t, uint64_t> counters() const {
    return {hits_.load(), misses_.load(), bytes_read_.load(), preads_.load()};
  }

  std::vector<std::tuple<int, int, int>> lru_items() {
    std::lock_guard<std::mutex> lk(state_mu_);
    std::vector<std::tuple<int, int, int>> out;
    for (auto& [key, slot] : order_) out.emplace_back(static_cast<int>(key / experts_), static_cast<int>(key % experts_), slot);
    return out;
  }

  std::vector<int> free_slots() {
    std::lock_guard<std::mutex> lk(state_mu_);
    return free_;
  }

  std::vector<int> window() {
    std::lock_guard<std::mutex> lk(state_mu_);
    return window_;
  }

  bool packed() const { return pack_fd_ >= 0; }
  int workers() const { return readers_ ? readers_->size() : 0; }

  // Serves the jobs already queued (a GPU signal pending past 5 s is abandoned), then stops every thread.
  void close() {
    {
      std::lock_guard<std::mutex> lk(jobs_mu_);
      if (closed_) return;
      closed_ = true;
      jobs_.push_back(Job{0, 0, 0, 0, 0, true});
    }
    jobs_cv_.notify_all();
    {
      std::unique_lock<std::mutex> lk(jobs_mu_);
      if (!jobs_cv_.wait_for(lk, std::chrono::seconds(5), [&] { return exited_; })) abort_ = true;
    }
    if (thread_.joinable()) thread_.join();
    readers_.reset();
    for (auto& buf : staging_) {
      std::free(buf);
      buf = nullptr;
    }
  }

 private:
  struct Job {
    int kind, layer, rows;
    uint64_t signal, ready;
    bool stop;
  };

  void serve() {
    while (true) {
      Job job;
      {
        std::unique_lock<std::mutex> lk(jobs_mu_);
        jobs_cv_.wait(lk, [&] { return !jobs_.empty(); });
        job = jobs_.front();
        jobs_.pop_front();
      }
      if (job.stop) break;
      try {
        wait_signal(job);
        std::lock_guard<std::mutex> lk(state_mu_);
        if (job.kind == 0) {
          size_t n = static_cast<size_t>(job.rows) * top_k_;
          if (job.rows < 0 || n * sizeof(uint32_t) > box_.nbytes()) {
            throw PoolError("IndexError", "peek: the array is not evaluated or the range passes its end");
          }
          const uint32_t* ids = box_.data<uint32_t>();
          std::vector<int64_t> routed(ids, ids + n);
          std::sort(routed.begin(), routed.end());
          routed.erase(std::unique(routed.begin(), routed.end()), routed.end());
          load(job.layer, routed);
        } else {
          const uint32_t* flags = box_.data<uint32_t>();
          std::vector<int64_t> present;
          for (int e = 0; e < experts_; ++e) {
            if (flags[e] != 0) present.push_back(e);
          }
          fill_window(job.layer, present);
        }
      } catch (const PoolError& e) {
        record(e.kind, e.what());
      } catch (const std::bad_alloc& e) {
        record("MemoryError", e.what());
      } catch (const std::exception& e) {
        record("RuntimeError", e.what());
      }
      channel_->signal(job.ready);   // the GPU always gets its release; the next Python call raises
    }
    std::lock_guard<std::mutex> lk(jobs_mu_);
    exited_ = true;
    jobs_cv_.notify_all();
  }

  void wait_signal(const Job& job) {
    auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms_);
    while (true) {
      if (channel_->event->waitUntilSignaledValue(job.signal, 100)) return;
      if (abort_.load()) throw PoolError("RuntimeError", "expert streaming closed while a call waited for the GPU");
      if (std::chrono::steady_clock::now() >= deadline) {
        throw PoolError("TimeoutError", "no routing from the GPU for layer " + std::to_string(job.layer));
      }
    }
  }

  void record(const std::string& kind, const std::string& msg) {
    std::lock_guard<std::mutex> lk(error_mu_);
    if (!error_) error_ = std::make_pair(kind, msg);
  }

  int64_t key(int layer, int64_t e) const { return static_cast<int64_t>(layer) * experts_ + e; }

  void lru_push(int64_t k, int slot) {
    order_.emplace_back(k, slot);
    where_[k] = std::prev(order_.end());
  }

  std::optional<int> lru_pop(int64_t k) {
    auto it = where_.find(k);
    if (it == where_.end()) return std::nullopt;
    int slot = it->second->second;
    order_.erase(it->second);
    where_.erase(it);
    return slot;
  }

  int evict(const std::unordered_set<int64_t>& keep) {
    for (auto it = order_.begin(); it != order_.end(); ++it) {
      if (!keep.count(it->first)) {
        int slot = it->second;
        where_.erase(it->first);
        order_.erase(it);
        return slot;
      }
    }
    throw PoolError("MemoryError", "the expert pool is smaller than one call's experts");
  }

  void check_layer(int layer) const {
    if (layer < 0 || layer >= layers_) throw PoolError("ValueError", "layer " + std::to_string(layer) + " is outside the model");
  }

  // Python's Streamer._load: every id resident in the LRU for ``layer``, then its slot written into the table.
  void load(int layer, const std::vector<int64_t>& ids) {
    check_layer(layer);
    std::unordered_set<int64_t> keep;
    for (int64_t e : ids) keep.insert(key(layer, e));
    std::vector<std::pair<int, int>> need, table;
    for (int64_t e : ids) {
      if (e < 0 || e >= experts_) {
        throw PoolError("ValueError", "routed expert " + std::to_string(e) + " is outside the model");
      }
      auto slot = lru_pop(key(layer, e));
      if (!slot) {
        if (!free_.empty()) {
          slot = free_.back();
          free_.pop_back();
        } else {
          slot = evict(keep);
        }
        need.emplace_back(static_cast<int>(e), *slot);
        misses_.fetch_add(1);
      } else {
        hits_.fetch_add(1);
      }
      lru_push(key(layer, e), *slot);
      table.emplace_back(static_cast<int>(e), *slot);
    }
    read_all(layer, need);
    int32_t* slot_of = slot_of_.data<int32_t>();
    for (auto [e, slot] : table) slot_of[static_cast<size_t>(layer) * experts_ + e] = slot;
  }

  void fill_window(int layer, const std::vector<int64_t>& ids) {
    check_layer(layer);
    std::vector<std::pair<int, int>> need;
    for (int64_t e : ids) {
      if (window_[e] != layer) need.emplace_back(static_cast<int>(e), static_cast<int>(e));
    }
    read_all(layer, need);
    for (auto [e, slot] : need) window_[e] = layer;
  }

  void read_all(int layer, const std::vector<std::pair<int, int>>& need) {
    if (need.empty()) return;
    size_t parts = dsts_.size();
    if (pack_fd_ >= 0) {
      readers_->run(need.size(), [&](size_t i, int worker) { read_packed(layer, need[i].first, need[i].second, worker); });
    } else {
      readers_->run(need.size() * parts, [&](size_t i, int) {
        read_part(layer, need[i / parts].first, i % parts, need[i / parts].second);
      });
    }
  }

  void read_part(int layer, int e, size_t p, int slot) {
    size_t at = (static_cast<size_t>(layer) * experts_ + e) * dsts_.size() + p;
    if (fds_[at] < 0) throw PoolError("KeyError", "no source for layer " + std::to_string(layer));
    pread_full(fds_[at], dst_ptrs_[p] + static_cast<size_t>(slot) * part_nbytes_[p], part_nbytes_[p], offsets_[at]);
    bytes_read_.fetch_add(part_nbytes_[p]);
    preads_.fetch_add(1);
  }

  void read_packed(int layer, int e, int slot, int worker) {
    int64_t offset = pack_offsets_[static_cast<size_t>(layer) * experts_ + e];
    if (offset < 0) throw PoolError("KeyError", "no packed record for layer " + std::to_string(layer));
    uint8_t*& buf = staging_[worker];
    if (buf == nullptr) {
      void* p = nullptr;
      if (posix_memalign(&p, pack_align_, pack_read_) != 0) throw std::bad_alloc();
      buf = static_cast<uint8_t*>(p);
    }
    pread_full(pack_fd_, buf, pack_read_, offset);
    preads_.fetch_add(1);
    for (size_t p = 0; p < dsts_.size(); ++p) {
      std::memcpy(dst_ptrs_[p] + static_cast<size_t>(slot) * part_nbytes_[p], buf + pack_parts_[p], part_nbytes_[p]);
    }
    bytes_read_.fetch_add(expert_nbytes_);
  }

  std::shared_ptr<Channel> channel_;
  mx::array box_, slot_of_;
  std::vector<mx::array> dsts_;
  std::vector<uint8_t*> dst_ptrs_;
  std::vector<size_t> part_nbytes_;
  size_t expert_nbytes_ = 0;
  std::vector<int32_t> fds_;
  std::vector<int64_t> offsets_;
  int layers_, experts_, top_k_, slots_;
  uint64_t timeout_ms_;

  int pack_fd_ = -1;
  std::vector<int64_t> pack_offsets_;
  size_t pack_read_ = 0, pack_align_ = 0;
  std::vector<size_t> pack_parts_;
  std::vector<uint8_t*> staging_;          // one aligned record buffer per reader thread

  std::mutex state_mu_;                    // the LRU, free list and window
  std::list<std::pair<int64_t, int>> order_;   // (layer * E + expert, slot), oldest first
  std::unordered_map<int64_t, std::list<std::pair<int64_t, int>>::iterator> where_;
  std::vector<int> free_, window_;

  std::atomic<uint64_t> hits_{0}, misses_{0}, bytes_read_{0}, preads_{0};
  std::mutex error_mu_;
  std::optional<std::pair<std::string, std::string>> error_;

  std::unique_ptr<Readers> readers_;
  std::mutex jobs_mu_;
  std::condition_variable jobs_cv_;
  std::deque<Job> jobs_;
  bool closed_ = false, exited_ = false;
  std::atomic<bool> abort_{false};
  std::thread thread_;
};

NB_MODULE(_hostsync, m) {
  nb::class_<Channel>(m, "Channel")
      .def(nb::init<>())
      .def("value", &Channel::value)
      .def("signal", &Channel::signal)
      .def("wait", &Channel::wait, nb::arg("value"), nb::arg("timeout_ms"));
  m.def("gpu_signal", [](const mx::array& x, std::shared_ptr<Channel> c, uint64_t v) { return mark(x, c, v, false); });
  m.def("gpu_wait", [](const mx::array& x, std::shared_ptr<Channel> c, uint64_t v) { return mark(x, c, v, true); });
  m.def("pread_into", &pread_into);
  m.def("write_into", &write_into);
  m.def("peek", &peek);
  nb::class_<PoolServer>(m, "PoolServer")
      .def(nb::init<std::shared_ptr<Channel>, mx::array, mx::array, std::vector<mx::array>, std::vector<size_t>,
                    nb::bytes, nb::bytes, int, int, int, int, int, uint64_t>(),
           nb::arg("channel"), nb::arg("box"), nb::arg("slot_of"), nb::arg("dsts"), nb::arg("part_nbytes"),
           nb::arg("fds"), nb::arg("offsets"), nb::arg("layers"), nb::arg("experts"), nb::arg("top_k"),
           nb::arg("slots"), nb::arg("workers"), nb::arg("timeout_ms"))
      .def("set_pack", &PoolServer::set_pack, nb::arg("fd"), nb::arg("offsets"), nb::arg("read_nbytes"),
           nb::arg("part_offsets"), nb::arg("alignment"))
      .def("submit", &PoolServer::submit, nb::arg("kind"), nb::arg("layer"), nb::arg("rows"), nb::arg("signal"),
           nb::arg("ready"))
      .def("error", &PoolServer::error)
      .def("counters", &PoolServer::counters)
      .def("lru_items", &PoolServer::lru_items)
      .def("free_slots", &PoolServer::free_slots)
      .def("window", &PoolServer::window)
      .def("packed", &PoolServer::packed)
      .def("workers", &PoolServer::workers)
      .def("close", &PoolServer::close, nb::call_guard<nb::gil_scoped_release>());
}
