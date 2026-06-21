'use strict';

const { describe, it, before, after } = require('node:test');
const assert = require('node:assert/strict');

const { CacheService } = require('../src/services/cacheService');

describe('CacheService', () => {
  let cache;

  before(() => {
    cache = new CacheService({ maxSize: 5, defaultTtlMs: 1000 });
  });

  after(() => {
    cache.destroy();
  });

  it('set and get a value', () => {
    cache.set('key1', 'value1');
    assert.equal(cache.get('key1'), 'value1');
  });

  it('returns undefined for missing key', () => {
    assert.equal(cache.get('nonexistent'), undefined);
  });

  it('respects TTL', async () => {
    cache.set('ttl-key', 'data', 50);
    assert.equal(cache.get('ttl-key'), 'data');
    await new Promise((r) => setTimeout(r, 80));
    assert.equal(cache.get('ttl-key'), undefined);
  });

  it('evicts oldest entry when full', () => {
    cache.clear();
    for (let i = 0; i < 5; i++) {
      cache.set(`fill-${i}`, i);
    }
    cache.set('overflow', 'new');
    assert.equal(cache.get('fill-0'), undefined);
    assert.equal(cache.get('overflow'), 'new');
  });

  it('delete removes entry', () => {
    cache.set('del-key', 'val');
    assert.equal(cache.get('del-key'), 'val');
    cache.delete('del-key');
    assert.equal(cache.get('del-key'), undefined);
  });

  it('invalidatePattern clears matching keys', () => {
    cache.clear();
    cache.set('user:1:name', 'Alice');
    cache.set('user:1:email', 'alice@test.com');
    cache.set('user:2:name', 'Bob');
    const count = cache.invalidatePattern('user:1');
    assert.equal(count, 2);
    assert.equal(cache.get('user:1:name'), undefined);
    assert.equal(cache.get('user:2:name'), 'Bob');
  });

  it('getStats tracks hits and misses', () => {
    cache.clear();
    cache.set('stat-key', 'val');
    cache.get('stat-key');
    cache.get('miss-key');
    const stats = cache.getStats();
    assert.equal(stats.hits, 1);
    assert.equal(stats.misses, 1);
  });

  it('getOrSet populates on miss', async () => {
    cache.clear();
    let called = 0;
    const val = await cache.getOrSet('compute', () => { called++; return 42; });
    assert.equal(val, 42);
    assert.equal(called, 1);

    const cached = await cache.getOrSet('compute', () => { called++; return 99; });
    assert.equal(cached, 42);
    assert.equal(called, 1);
  });
});
