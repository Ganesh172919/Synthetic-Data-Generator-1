'use strict';

const { generateApiKey, hashApiKey, generateId } = require('../utils/crypto');
const { NotFoundError, ValidationError, QuotaExceededError } = require('../utils/errors');
const { nowIso, safeJsonParse } = require('../db');

class ApiKeyService {
  constructor(db) {
    this.db = db;

    this.stmts = {
      findById: db.prepare('SELECT * FROM api_keys WHERE id = ?'),
      findByHash: db.prepare('SELECT * FROM api_keys WHERE key_hash = ? AND is_active = 1'),
      listByUser: db.prepare('SELECT * FROM api_keys WHERE user_id = ? ORDER BY datetime(created_at) DESC'),
      countByUser: db.prepare('SELECT COUNT(*) AS count FROM api_keys WHERE user_id = ? AND is_active = 1'),
      insert: db.prepare(`
        INSERT INTO api_keys (id, user_id, name, key_hash, key_prefix, scopes, is_active, created_at)
        VALUES (?, ?, ?, ?, ?, ?, 1, ?)
      `),
      deactivate: db.prepare('UPDATE api_keys SET is_active = 0 WHERE id = ? AND user_id = ?'),
      updateLastUsed: db.prepare('UPDATE api_keys SET last_used_at = ? WHERE id = ?'),
      deleteKey: db.prepare('DELETE FROM api_keys WHERE id = ? AND user_id = ?'),
    };
  }

  _toApi(row) {
    if (!row) return null;
    return {
      id: row.id,
      userId: row.user_id,
      name: row.name,
      keyPrefix: row.key_prefix,
      scopes: safeJsonParse(row.scopes, []),
      isActive: Boolean(row.is_active),
      lastUsedAt: row.last_used_at,
      expiresAt: row.expires_at,
      createdAt: row.created_at,
    };
  }

  /**
   * Create a new API key. Returns the raw key only once.
   */
  create(userId, { name, scopes = ['read', 'write'], maxKeysAllowed = 10 }) {
    if (!name || typeof name !== 'string' || name.trim().length < 1) {
      throw new ValidationError('API key name is required');
    }

    const activeCount = this.stmts.countByUser.get(userId).count;
    if (activeCount >= maxKeysAllowed) {
      throw new QuotaExceededError(`API key limit (${maxKeysAllowed} keys)`);
    }

    const rawKey = generateApiKey();
    const keyHash = hashApiKey(rawKey);
    const keyPrefix = rawKey.slice(0, 12) + '...';
    const id = generateId('key');
    const scopesJson = JSON.stringify(scopes);

    this.stmts.insert.run(id, userId, name.trim(), keyHash, keyPrefix, scopesJson, nowIso());

    return {
      ...this._toApi(this.stmts.findById.get(id)),
      rawKey,
    };
  }

  /**
   * Validate a raw API key and return the user and key info.
   */
  validateKey(rawKey) {
    if (!rawKey || typeof rawKey !== 'string') return null;

    const keyHash = hashApiKey(rawKey);
    const row = this.stmts.findByHash.get(keyHash);
    if (!row) return null;

    if (row.expires_at && new Date(row.expires_at) < new Date()) {
      return null;
    }

    this.stmts.updateLastUsed.run(nowIso(), row.id);

    return {
      keyId: row.id,
      userId: row.user_id,
      scopes: safeJsonParse(row.scopes, []),
    };
  }

  /**
   * List all API keys for a user (without showing the actual key).
   */
  listByUser(userId) {
    const rows = this.stmts.listByUser.all(userId);
    return rows.map((r) => this._toApi(r));
  }

  /**
   * Revoke (deactivate) an API key.
   */
  revoke(keyId, userId) {
    const result = this.stmts.deactivate.run(keyId, userId);
    if (result.changes === 0) {
      throw new NotFoundError('API key');
    }
    return true;
  }

  /**
   * Permanently delete an API key.
   */
  deleteKey(keyId, userId) {
    const result = this.stmts.deleteKey.run(keyId, userId);
    if (result.changes === 0) {
      throw new NotFoundError('API key');
    }
    return true;
  }
}

module.exports = { ApiKeyService };
