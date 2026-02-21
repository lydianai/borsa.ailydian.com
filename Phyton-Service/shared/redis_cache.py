"""
🗄️ REDIS CACHE HELPER
=====================
Centralized Redis caching for all services

Features:
- Automatic JSON serialization
- TTL management
- Cache invalidation
- Graceful fallback (works without Redis)
- Pattern-based deletion
"""

import json
from typing import Optional, Any, List
import hashlib


class RedisCache:
    """Redis cache wrapper with graceful fallback"""

    def __init__(
        self,
        host: str = 'localhost',
        port: int = 6379,
        db: int = 0,
        password: Optional[str] = None,
        enabled: bool = True
    ):
        self.enabled = enabled
        self.client = None

        if self.enabled:
            try:
                import redis
                self.client = redis.Redis(
                    host=host,
                    port=port,
                    db=db,
                    password=password,
                    decode_responses=True,
                    socket_connect_timeout=2,
                    socket_timeout=2
                )
                # Test connection
                self.client.ping()
                print(f"✅ Redis connected: {host}:{port}")
            except Exception as e:
                print(f"⚠️  Redis not available, caching disabled: {e}")
                self.client = None
                self.enabled = False

    def _make_key(self, prefix: str, key: str) -> str:
        """Generate cache key with prefix"""
        return f"{prefix}:{key}"

    def _serialize(self, value: Any) -> str:
        """Serialize value to JSON string"""
        return json.dumps(value, default=str)

    def _deserialize(self, value: str) -> Any:
        """Deserialize JSON string to value"""
        try:
            return json.loads(value)
        except:
            return value

    def get(self, prefix: str, key: str) -> Optional[Any]:
        """
        Get value from cache

        Args:
            prefix: Cache prefix (e.g., "price", "indicator")
            key: Cache key (e.g., "BTCUSDT")

        Returns:
            Cached value or None if not found
        """
        if not self.enabled or not self.client:
            return None

        try:
            cache_key = self._make_key(prefix, key)
            data = self.client.get(cache_key)
            if data:
                return self._deserialize(data)
        except Exception as e:
            print(f"⚠️  Cache get error: {e}")
        return None

    def set(
        self,
        prefix: str,
        key: str,
        value: Any,
        ttl: int = 60
    ) -> bool:
        """
        Set value in cache with TTL

        Args:
            prefix: Cache prefix
            key: Cache key
            value: Value to cache (will be JSON serialized)
            ttl: Time to live in seconds (default: 60s)

        Returns:
            True if successful, False otherwise
        """
        if not self.enabled or not self.client:
            return False

        try:
            cache_key = self._make_key(prefix, key)
            serialized = self._serialize(value)
            self.client.setex(cache_key, ttl, serialized)
            return True
        except Exception as e:
            print(f"⚠️  Cache set error: {e}")
            return False

    def delete(self, prefix: str, key: str) -> bool:
        """Delete specific cache entry"""
        if not self.enabled or not self.client:
            return False

        try:
            cache_key = self._make_key(prefix, key)
            self.client.delete(cache_key)
            return True
        except Exception as e:
            print(f"⚠️  Cache delete error: {e}")
            return False

    def delete_pattern(self, pattern: str) -> int:
        """
        Delete all keys matching pattern

        Args:
            pattern: Redis pattern (e.g., "price:*", "*USDT")

        Returns:
            Number of keys deleted
        """
        if not self.enabled or not self.client:
            return 0

        try:
            keys = self.client.keys(pattern)
            if keys:
                return self.client.delete(*keys)
        except Exception as e:
            print(f"⚠️  Cache delete pattern error: {e}")
        return 0

    def clear_all(self) -> bool:
        """Clear all cache (use with caution!)"""
        if not self.enabled or not self.client:
            return False

        try:
            self.client.flushdb()
            return True
        except Exception as e:
            print(f"⚠️  Cache clear error: {e}")
            return False

    def get_ttl(self, prefix: str, key: str) -> Optional[int]:
        """Get remaining TTL for a key"""
        if not self.enabled or not self.client:
            return None

        try:
            cache_key = self._make_key(prefix, key)
            ttl = self.client.ttl(cache_key)
            return ttl if ttl > 0 else None
        except Exception as e:
            print(f"⚠️  Cache TTL error: {e}")
            return None

    def exists(self, prefix: str, key: str) -> bool:
        """Check if key exists in cache"""
        if not self.enabled or not self.client:
            return False

        try:
            cache_key = self._make_key(prefix, key)
            return bool(self.client.exists(cache_key))
        except Exception as e:
            print(f"⚠️  Cache exists error: {e}")
            return False

    def get_stats(self) -> dict:
        """Get cache statistics"""
        if not self.enabled or not self.client:
            return {"enabled": False, "status": "disabled"}

        try:
            info = self.client.info()
            return {
                "enabled": True,
                "status": "connected",
                "used_memory": info.get("used_memory_human", "N/A"),
                "total_keys": self.client.dbsize(),
                "uptime_seconds": info.get("uptime_in_seconds", 0)
            }
        except Exception as e:
            return {"enabled": False, "status": "error", "error": str(e)}


def hash_key(data: Any) -> str:
    """
    Generate cache key from complex data (e.g., function arguments)

    Args:
        data: Any data (dict, list, string, etc.)

    Returns:
        MD5 hash string
    """
    serialized = json.dumps(data, sort_keys=True, default=str)
    return hashlib.md5(serialized.encode()).hexdigest()


# Example usage
if __name__ == "__main__":
    # Create cache instance
    cache = RedisCache(enabled=True)

    # Test basic operations
    print("\n📝 Testing basic operations...")
    cache.set("test", "key1", {"price": 100, "symbol": "BTC"}, ttl=10)
    result = cache.get("test", "key1")
    print(f"Cached data: {result}")

    # Test existence
    exists = cache.exists("test", "key1")
    print(f"Key exists: {exists}")

    # Test TTL
    ttl = cache.get_ttl("test", "key1")
    print(f"Remaining TTL: {ttl}s")

    # Test stats
    stats = cache.get_stats()
    print(f"\nCache stats: {json.dumps(stats, indent=2)}")

    # Test hash key
    complex_data = {"symbol": "BTCUSDT", "interval": "1h", "indicators": ["RSI", "MACD"]}
    key = hash_key(complex_data)
    print(f"\nHashed key: {key}")

    # Cleanup
    cache.delete("test", "key1")

    print("\n✅ Redis cache test completed")
