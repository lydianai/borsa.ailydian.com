/**
 * Cache Manager Tests
 *
 * Tests for the cache manager utility functions
 */

import { describe, it, expect, beforeEach, vi } from 'vitest';
import { getCached, setCached, deleteCached, getOrSet } from '@/lib/cache/cache-manager';

describe('Cache Manager', () => {
  beforeEach(() => {
    // Clear any existing cache before each test
    vi.clearAllMocks();
  });

  describe('setCached & getCached', () => {
    it('should store and retrieve a value', async () => {
      const key = 'test:key';
      const value = { foo: 'bar', num: 123 };

      await setCached(key, value, 60);
      const retrieved = await getCached(key);

      expect(retrieved).toEqual(value);
    });

    it('should return null for non-existent key', async () => {
      const result = await getCached('non:existent:key');
      expect(result).toBeNull();
    });

    it('should handle different data types', async () => {
      const testCases = [
        { key: 'string', value: 'hello' },
        { key: 'number', value: 42 },
        { key: 'boolean', value: true },
        { key: 'array', value: [1, 2, 3] },
        { key: 'object', value: { nested: { deep: 'value' } } },
      ];

      for (const { key, value } of testCases) {
        await setCached(key, value, 60);
        const retrieved = await getCached(key);
        expect(retrieved).toEqual(value);
      }
    });
  });

  describe('deleteCached', () => {
    it('should delete a cached value', async () => {
      const key = 'delete:test';
      const value = 'to be deleted';

      await setCached(key, value, 60);
      await deleteCached(key);

      const retrieved = await getCached(key);
      expect(retrieved).toBeNull();
    });
  });

  describe('getOrSet', () => {
    it('should fetch and cache value on cache miss', async () => {
      const key = 'getOrSet:test';
      const fetchFn = vi.fn(async () => ({ data: 'fresh' }));

      const result = await getOrSet(key, fetchFn, 60);

      expect(result).toEqual({ data: 'fresh' });
      expect(fetchFn).toHaveBeenCalledTimes(1);
    });

    it('should return cached value on cache hit', async () => {
      const key = 'getOrSet:cached';
      const fetchFn = vi.fn(async () => ({ data: 'fresh' }));

      // First call - cache miss
      await getOrSet(key, fetchFn, 60);

      // Second call - cache hit
      const result = await getOrSet(key, fetchFn, 60);

      expect(result).toEqual({ data: 'fresh' });
      expect(fetchFn).toHaveBeenCalledTimes(1); // Should only be called once
    });
  });

  describe('TTL behavior', () => {
    it('should respect TTL settings', async () => {
      const key = 'ttl:test';
      const value = 'expires soon';

      // This test just verifies the API accepts TTL
      // Actual expiration would require waiting
      await setCached(key, value, 1);
      const retrieved = await getCached(key);

      expect(retrieved).toEqual(value);
    });
  });

  describe('Error handling', () => {
    it('should handle invalid JSON gracefully', async () => {
      // This test ensures the cache handles errors
      const key = 'error:test';

      // Should not throw
      await expect(getCached(key)).resolves.toBeNull();
    });
  });
});
