'use strict';

/**
 * In-memory LRU cache with TTL support.
 * Designed for caching database query results, computed metrics, and API responses.
 */

class CacheEntry {
  constructor(value, ttlMs) {
    this.value = value;
    this.expiresAt = Date.now() + ttlMs;
  }

  isExpired() {
    return Date.now() > this.expiresAt;
  }
}

class CacheService {
  constructor({ maxSize = 1000, defaultTtlMs = 60000 } = {}) {
    this.maxSize = maxSize;
    this.defaultTtlMs = defaultTtlMs;
    this.store = new Map();
    this.hits = 0;
    this.misses = 0;

    this._cleanupInterval = setInterval(() => this.evictExpired(), 30000);
    if (this._cleanupInterval.unref) {
      this._cleanupInterval.unref();
    }
  }

  get(key) {
    const entry = this.store.get(key);
    if (!entry || entry.isExpired()) {
      if (entry) {
        this.store.delete(key);
      }
      this.misses++;
      return undefined;
    }

    this.store.delete(key);
    this.store.set(key, entry);
    this.hits++;
    return entry.value;
  }

  set(key, value, ttlMs = null) {
    if (this.store.has(key)) {
      this.store.delete(key);
    }

    if (this.store.size >= this.maxSize) {
      const firstKey = this.store.keys().next().value;
      this.store.delete(firstKey);
    }

    this.store.set(key, new CacheEntry(value, ttlMs || this.defaultTtlMs));
  }

  delete(key) {
    return this.store.delete(key);
  }

  invalidatePattern(pattern) {
    let count = 0;
    for (const key of this.store.keys()) {
      if (key.startsWith(pattern) || key.includes(pattern)) {
        this.store.delete(key);
        count++;
      }
    }
    return count;
  }

  clear() {
    this.store.clear();
    this.hits = 0;
    this.misses = 0;
  }

  evictExpired() {
    for (const [key, entry] of this.store.entries()) {
      if (entry.isExpired()) {
        this.store.delete(key);
      }
    }
  }

  getStats() {
    const total = this.hits + this.misses;
    return {
      size: this.store.size,
      maxSize: this.maxSize,
      hits: this.hits,
      misses: this.misses,
      hitRate: total > 0 ? (this.hits / total * 100).toFixed(2) + '%' : '0%',
    };
  }

  /**
   * Get-or-set pattern: returns cached value or calls fn() to populate.
   */
  async getOrSet(key, fn, ttlMs = null) {
    const cached = this.get(key);
    if (cached !== undefined) {
      return cached;
    }
    const value = await fn();
    this.set(key, value, ttlMs);
    return value;
  }

  destroy() {
    clearInterval(this._cleanupInterval);
    this.clear();
  }
}

const cache = new CacheService({ maxSize: 2000, defaultTtlMs: 60000 });

module.exports = { CacheService, cache };
