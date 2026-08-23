from functools import lru_cache

from langchain_openai import ChatOpenAI

from inquirygraph.config.settings import settings


@lru_cache
def get_main_llm() -> ChatOpenAI:
    """Planning, synthesis, coverage decisions — higher-quality model."""
    return ChatOpenAI(
        model=settings.llm_main_model,
        api_key=settings.openrouter_api_key,
        base_url=settings.openrouter_base_url,
        temperature=0.2,
        timeout=90,
        max_retries=0,
    )


@lru_cache
def get_fast_llm() -> ChatOpenAI:
    """Extraction, query generation — cheaper/faster model."""
    return ChatOpenAI(
        model=settings.llm_fast_model,
        api_key=settings.openrouter_api_key,
        base_url=settings.openrouter_base_url,
        temperature=0.0,
        timeout=30,
        max_retries=0,
    )
