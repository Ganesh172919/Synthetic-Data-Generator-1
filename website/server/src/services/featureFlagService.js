'use strict';

const { NotFoundError, FeatureDisabledError } = require('../utils/errors');
const { safeJsonParse, nowIso } = require('../db');
const { cache } = require('./cacheService');

class FeatureFlagService {
  constructor(db) {
    this.db = db;

    this.stmts = {
      findByName: db.prepare('SELECT * FROM feature_flags WHERE name = ?'),
      findById: db.prepare('SELECT * FROM feature_flags WHERE id = ?'),
      listAll: db.prepare('SELECT * FROM feature_flags ORDER BY name'),
      update: db.prepare(`
        UPDATE feature_flags SET is_enabled = ?, allowed_tiers = ?, config_json = ?, updated_at = ?
        WHERE name = ?
      `),
      toggle: db.prepare('UPDATE feature_flags SET is_enabled = ?, updated_at = ? WHERE name = ?'),
    };
  }

  _toApi(row) {
    if (!row) return null;
    return {
      id: row.id,
      name: row.name,
      description: row.description,
      isEnabled: Boolean(row.is_enabled),
      allowedTiers: safeJsonParse(row.allowed_tiers, []),
      allowedRoles: safeJsonParse(row.allowed_roles, []),
      config: safeJsonParse(row.config_json, null),
      createdAt: row.created_at,
      updatedAt: row.updated_at,
    };
  }

  getFlag(name) {
    const cacheKey = `ff:${name}`;
    const cached = cache.get(cacheKey);
    if (cached) return cached;

    const row = this.stmts.findByName.get(name);
    if (!row) return null;

    const flag = this._toApi(row);
    cache.set(cacheKey, flag, 30000);
    return flag;
  }

  listAll() {
    return this.stmts.listAll.all().map((r) => this._toApi(r));
  }

  /**
   * Check if a feature is enabled for a specific tier and role.
   * Throws FeatureDisabledError if not accessible.
   */
  checkAccess(featureName, tier = 'free', role = 'user') {
    const flag = this.getFlag(featureName);

    if (!flag) {
      return true;
    }

    if (!flag.isEnabled) {
      throw new FeatureDisabledError(featureName);
    }

    if (!flag.allowedTiers.includes(tier)) {
      throw new FeatureDisabledError(featureName);
    }

    if (!flag.allowedRoles.includes(role)) {
      throw new FeatureDisabledError(featureName);
    }

    return true;
  }

  /**
   * Check if a feature is available (non-throwing version).
   */
  isEnabled(featureName, tier = 'free', role = 'user') {
    try {
      return this.checkAccess(featureName, tier, role);
    } catch {
      return false;
    }
  }

  /**
   * Toggle a feature flag on/off (admin only).
   */
  toggle(name, enabled) {
    const row = this.stmts.findByName.get(name);
    if (!row) throw new NotFoundError('Feature flag');

    this.stmts.toggle.run(enabled ? 1 : 0, nowIso(), name);
    cache.delete(`ff:${name}`);
    return this.getFlag(name);
  }

  /**
   * Update a feature flag configuration (admin only).
   */
  updateFlag(name, { isEnabled, allowedTiers, config }) {
    const row = this.stmts.findByName.get(name);
    if (!row) throw new NotFoundError('Feature flag');

    const enabled = isEnabled !== undefined ? (isEnabled ? 1 : 0) : row.is_enabled;
    const tiers = allowedTiers ? JSON.stringify(allowedTiers) : row.allowed_tiers;
    const configJson = config !== undefined ? JSON.stringify(config) : row.config_json;

    this.stmts.update.run(enabled, tiers, configJson, nowIso(), name);
    cache.delete(`ff:${name}`);
    return this.getFlag(name);
  }
}

module.exports = { FeatureFlagService };
