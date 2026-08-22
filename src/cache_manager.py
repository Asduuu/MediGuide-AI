"""
cache_manager.py
LangChain Cache Configuration for MediGuide AI

Demonstrates InMemoryCache and SQLiteCache implementations to optimize performance,
reduce cost, and eliminate redundant LLM API calls for identical patient inputs.
"""

import os
from typing import Tuple

# LangChain Global Cache imports
try:
    from langchain_core.globals import set_llm_cache
except ImportError:
    from langchain.globals import set_llm_cache

try:
    from langchain_community.cache import InMemoryCache, SQLiteCache
except ImportError:
    from langchain.cache import InMemoryCache, SQLiteCache

DB_FILE_PATH = ".langchain.db"


def configure_cache(cache_type: str) -> Tuple[bool, str]:
    """
    Configures the global LangChain LLM cache based on user selection.

    Args:
        cache_type (str): "Disabled", "InMemoryCache", or "SQLiteCache"

    Returns:
        Tuple[bool, str]: (Success status, Human-readable status message)
    """
    try:
        if cache_type == "InMemoryCache":
            set_llm_cache(InMemoryCache())
            return True, "⚡ **InMemoryCache** active (RAM-based caching enabled)."
        
        elif cache_type == "SQLiteCache":
            set_llm_cache(SQLiteCache(database_path=DB_FILE_PATH))
            return True, f"💾 **SQLiteCache** active (Disk-backed database caching at `{DB_FILE_PATH}`)."
        
        else:
            set_llm_cache(None)
            return True, "🛑 Caching disabled (Every request hits OpenAI API)."

    except Exception as e:
        set_llm_cache(None)
        return False, f"⚠️ Cache configuration error: {str(e)}"


def get_cache_info(cache_type: str) -> dict:
    """
    Returns metadata explaining the active caching strategy.
    """
    if cache_type == "InMemoryCache":
        return {
            "name": "InMemoryCache",
            "storage": "RAM (Volatile)",
            "persistence": "No (cleared when app restarts)",
            "speed": "Ultra-fast (< 5ms)",
            "best_for": "Single interactive session testing"
        }
    elif cache_type == "SQLiteCache":
        db_exists = os.path.exists(DB_FILE_PATH)
        db_size = os.path.getsize(DB_FILE_PATH) if db_exists else 0
        return {
            "name": "SQLiteCache",
            "storage": f"Disk File (`{DB_FILE_PATH}` - {db_size / 1024:.1f} KB)",
            "persistence": "Yes (survives app restarts)",
            "speed": "Fast (< 20ms)",
            "best_for": "Cross-session persistent caching"
        }
    else:
        return {
            "name": "Disabled",
            "storage": "None",
            "persistence": "N/A",
            "speed": "Standard API Latency (1-3 sec)",
            "best_for": "Always fetching fresh LLM responses"
        }
