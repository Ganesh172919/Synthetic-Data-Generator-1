'use strict';

const { describe, it } = require('node:test');
const assert = require('node:assert/strict');

const {
  hashPassword,
  verifyPassword,
  generateApiKey,
  hashApiKey,
  signJwt,
  verifyJwt,
  generateId,
} = require('../src/utils/crypto');

describe('crypto utilities', () => {
  describe('hashPassword / verifyPassword', () => {
    it('hashes and verifies a password', () => {
      const hash = hashPassword('my-secret');
      assert.ok(hash.includes(':'));
      assert.ok(verifyPassword('my-secret', hash));
    });

    it('rejects wrong password', () => {
      const hash = hashPassword('correct');
      assert.equal(verifyPassword('wrong', hash), false);
    });

    it('produces unique hashes for same password', () => {
      const h1 = hashPassword('same');
      const h2 = hashPassword('same');
      assert.notEqual(h1, h2);
    });
  });

  describe('generateApiKey / hashApiKey', () => {
    it('generates key with prefix', () => {
      const key = generateApiKey('sg_test');
      assert.ok(key.startsWith('sg_test_'));
      assert.ok(key.length > 20);
    });

    it('hashApiKey produces consistent hash', () => {
      const key = generateApiKey();
      const h1 = hashApiKey(key);
      const h2 = hashApiKey(key);
      assert.equal(h1, h2);
    });

    it('different keys produce different hashes', () => {
      const k1 = generateApiKey();
      const k2 = generateApiKey();
      assert.notEqual(hashApiKey(k1), hashApiKey(k2));
    });
  });

  describe('signJwt / verifyJwt', () => {
    const secret = 'test-secret-key';

    it('signs and verifies a token', () => {
      const token = signJwt({ sub: 'user123', role: 'admin' }, secret, 3600);
      const payload = verifyJwt(token, secret);
      assert.ok(payload);
      assert.equal(payload.sub, 'user123');
      assert.equal(payload.role, 'admin');
      assert.ok(payload.iat);
      assert.ok(payload.exp);
    });

    it('rejects token with wrong secret', () => {
      const token = signJwt({ sub: 'user123' }, secret);
      assert.equal(verifyJwt(token, 'wrong-secret'), null);
    });

    it('rejects expired token', () => {
      const token = signJwt({ sub: 'user123' }, secret, -1);
      assert.equal(verifyJwt(token, secret), null);
    });

    it('rejects malformed token', () => {
      assert.equal(verifyJwt('not.a.jwt.token', secret), null);
      assert.equal(verifyJwt('', secret), null);
      assert.equal(verifyJwt('abc', secret), null);
    });
  });

  describe('generateId', () => {
    it('generates with prefix', () => {
      const id = generateId('usr');
      assert.ok(id.startsWith('usr_'));
    });

    it('generates without prefix', () => {
      const id = generateId();
      assert.ok(id.length > 0);
      assert.ok(!id.includes('_'));
    });

    it('generates unique IDs', () => {
      const ids = new Set(Array.from({ length: 100 }, () => generateId('test')));
      assert.equal(ids.size, 100);
    });
  });
});
