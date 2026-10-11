"""Page-cache residency and no-sudo eviction of a model's checkpoint files: see tensorfold.streaming.page_cache.

    python tools/page_cache.py resident MODEL_DIR [--experts]
    python tools/page_cache.py evict MODEL_DIR"""

import sys

from tensorfold.streaming.pack import PackError
from tensorfold.streaming.page_cache import main

if __name__ == "__main__":
    try:
        sys.exit(main())
    except (PackError, FileNotFoundError, KeyError, ValueError) as error:   # --experts on a model without SSD experts
        print(f"page_cache: {type(error).__name__}: {error}", file=sys.stderr)
        sys.exit(2)
