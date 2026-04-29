"""pytest 부트스트랩 — src/를 sys.path 루트로 등록.

런타임(uvicorn `--app-dir src`, celery `working_dir=/app/src`)이 src/를 import 루트로
사용하므로 pytest도 동일하게 맞춤. 그래야:

  1. api.py 의 `from celery_app import ...`, `from v1.router import ...` 가 컨테이너·호스트
     양쪽에서 동일 module 이름으로 해결 (try/except fallback 불필요).
  2. 통합 테스트의 `@patch("v1.router.X")` 가 api.py 가 실제 import한 그 module instance
     에 적중 (mock binding 일치).

`pyproject.toml [tool.pytest.ini_options].pythonpath` 로도 가능하지만,
docker volume 마운트 누락 / 컨테이너 재생성 누락 같은 운영 사고에 취약해 conftest.py 로 고정.
tests/ 는 항상 마운트되어 있어 host 변경 즉시 컨테이너에 반영.
"""
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
