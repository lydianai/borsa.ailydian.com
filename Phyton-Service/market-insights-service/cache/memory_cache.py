"""
SIMPLE MEMORY CACHE
Caches data for 5 minutes to reduce API calls
"""

import time
from typing import Dict, Optional

cache_storage = {}


def get_cache(key: str) -> Optional[Dict]:
    """Get cached data if not expired"""
    if key in cache_storage:
        data, timestamp = cache_storage[key]
        if time.time() - timestamp < 300:  # 5 minutes TTL
            return data
        else:
            del cache_storage[key]
    return None


def set_cache(key: str, data: Dict):
    """Set cache with current timestamp"""
    cache_storage[key] = (data, time.time())


def clear_cache():
    """Clear all cache"""
    cache_storage.clear()
