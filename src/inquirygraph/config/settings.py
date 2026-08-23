from enum import Enum

from pydantic_settings import BaseSettings, SettingsConfigDict


class ProcessingMode(str, Enum):
    SYNC = "sync"
    ASYNC = "async"


class RetrievalStrategy(str, Enum):
    VECTOR = "vector"
    HYBRID = "hybrid"
    GRAPH = "graph"
    GRAPH_PLUS_VECTOR = "graph_plus_vector"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    openrouter_api_key: str = ""
    openrouter_base_url: str = "https://openrouter.ai/api/v1"

    llm_main_model: str = "deepseek/deepseek-v4-flash-0731"
    llm_fast_model: str = "qwen/qwen3-30b-a3b-instruct-2507"

    tavily_api_key: str = ""

    postgres_url: str = "postgresql://inquiry:inquiry@localhost:5432/inquirygraph"
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "inquirygraph"

    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "evidence_chunks"

    max_research_iterations: int = 2
    max_web_results: int = 5
    max_sources_per_task: int = 1
    max_total_sources: int = 1
    max_tasks_per_plan: int = 4
    enable_reranking: bool = False
    retrieval_strategy: RetrievalStrategy = RetrievalStrategy.HYBRID

    api_host: str = "0.0.0.0"
    api_port: int = 8000
    processing_mode: ProcessingMode = ProcessingMode.ASYNC

    checkpoint_db_path: str = "checkpoints.db"


settings = Settings()
