"""청킹 전략 선택.
CHUNKER_TYPE 환경변수로 adaptive(헤딩 트리) / fixed(윈도우 슬라이딩) 전환.
"""

import os

from .chunker_adaptive import Chunk, to_json  # noqa: F401

CHUNKER_TYPE = os.environ["CHUNKER_TYPE"]

if CHUNKER_TYPE == "fixed":
    from .chunker_fixed import chunk_markdown  # noqa: F401
else:
    from .chunker_adaptive import chunk_markdown  # noqa: F401
