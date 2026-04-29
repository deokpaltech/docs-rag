"""BGE-M3 임베딩.
local_files_only=True로 HF Hub 접속 없이 로컬 모델만 사용.
싱글턴으로 로드하여 태스크 간 모델 재로딩 방지.
"""

from sentence_transformers import SentenceTransformer

from ..config import EMBEDDING_CONFIG

_model = None


def get_embedding_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_CONFIG["model_path"], local_files_only=True)
    return _model


def embed_texts(texts: list[str]) -> list[list[float]]:
    model = get_embedding_model()
    return model.encode(texts, normalize_embeddings=True).tolist()


def embed_query(query: str) -> list[float]:
    return embed_texts([query])[0]


def count_tokens(texts: list[str]) -> list[int]:
    """BGE-M3 토크나이저 기준 토큰 수 계산."""
    tokenizer = get_embedding_model().tokenizer
    return [len(tokenizer.encode(t, add_special_tokens=False)) for t in texts]
