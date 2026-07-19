from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

SCAFFOLD_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=SCAFFOLD_ROOT / ".env", extra="ignore")

    # LLM
    openai_api_key: str = ""
    openai_model: str = "gpt-5.6-luna"
    openai_embed_model: str = "text-embedding-3-small"
    openai_model_input_price_per_1m: float = 0.0
    openai_model_output_price_per_1m: float = 0.0
    openai_embed_price_per_1m: float = 0.0

    # Pinecone
    pinecone_api_key: str = ""
    pinecone_index: str = "knowledge-assistant"
    pinecone_cloud: str = "aws"
    pinecone_region: str = "us-east-1"
    embed_dimension: int = 1536

    # Retrieval
    top_k: int = 8
    score_floor: float = 0.25

    # Agent
    compressor_concurrency: int = 8

    # Paths
    data_dir: Path = SCAFFOLD_ROOT / "data"
    log_dir: Path = SCAFFOLD_ROOT / "logs"
    question_bank_dir: Path = SCAFFOLD_ROOT / "eval" / "question_bank"
    eval_runs_dir: Path = SCAFFOLD_ROOT / "eval" / "runs"
    ingest_state_file: Path = SCAFFOLD_ROOT / ".ingest_state.json"

    @property
    def users_file(self) -> Path:
        return self.data_dir / "users.json"

    @property
    def manifest_file(self) -> Path:
        return self.data_dir / "pdfs" / "manifest.json"

    @property
    def pdf_dir(self) -> Path:
        return self.data_dir / "pdfs"


@lru_cache
def get_settings() -> Settings:
    return Settings()
