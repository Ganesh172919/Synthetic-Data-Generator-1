'use strict';

const { generateId } = require('../utils/crypto');
const { NotFoundError, ValidationError, QuotaExceededError } = require('../utils/errors');
const { nowIso } = require('../db');

/**
 * Subscription tier definitions with limits and feature entitlements.
 * These are the source of truth for tier-based gating.
 */
const TIER_DEFINITIONS = {
  free: {
    name: 'Free',
    price: 0,
    limits: {
      jobsPerDay: 5,
      maxTargetCount: 1000,
      maxBatchSize: 25,
      maxConcurrentJobs: 1,
      apiKeysAllowed: 1,
      retentionDays: 3,
      maxDomainsCount: 2,
    },
    features: ['csv_export', 'json_export'],
    providers: ['mock'],
  },
  pro: {
    name: 'Pro',
    price: 29,
    limits: {
      jobsPerDay: 50,
      maxTargetCount: 50000,
      maxBatchSize: 50,
      maxConcurrentJobs: 3,
      apiKeysAllowed: 10,
      retentionDays: 30,
      maxDomainsCount: 20,
    },
    features: [
      'csv_export', 'json_export', 'custom_domains', 'api_access',
      'bulk_generation', 'advanced_analytics', 'plugin_marketplace',
    ],
    providers: ['mock', 'huggingface', 'openai'],
  },
  enterprise: {
    name: 'Enterprise',
    price: 199,
    limits: {
      jobsPerDay: 500,
      maxTargetCount: 100000,
      maxBatchSize: 50,
      maxConcurrentJobs: 10,
      apiKeysAllowed: 100,
      retentionDays: 90,
      maxDomainsCount: 100,
    },
    features: [
      'csv_export', 'json_export', 'custom_domains', 'api_access',
      'bulk_generation', 'advanced_analytics', 'plugin_marketplace',
      'priority_queue', 'team_management', 'sla_support',
    ],
    providers: ['mock', 'huggingface', 'openai'],
  },
};

class SubscriptionService {
  constructor(db) {
    this.db = db;
    this.tiers = TIER_DEFINITIONS;

    this.stmts = {
      findByUserId: db.prepare('SELECT * FROM subscriptions WHERE user_id = ?'),
      insert: db.prepare(`
        INSERT INTO subscriptions (id, user_id, tier, status, started_at, expires_at, created_at, updated_at)
        VALUES (?, ?, ?, 'active', ?, ?, ?, ?)
      `),
      updateTier: db.prepare(`
        UPDATE subscriptions SET tier = ?, status = 'active', updated_at = ? WHERE user_id = ?
      `),
      cancel: db.prepare(`
        UPDATE subscriptions SET status = 'cancelled', cancelled_at = ?, updated_at = ? WHERE user_id = ?
      `),
      countJobsToday: db.prepare(`
        SELECT COUNT(*) AS count FROM jobs
        WHERE user_id = ? AND created_at >= ?
      `),
      countActiveDomains: db.prepare(`
        SELECT COUNT(*) AS count FROM domains
      `),
    };
  }

  getTierDefinition(tier) {
    return this.tiers[tier] || this.tiers.free;
  }

  getAllTiers() {
    return Object.entries(this.tiers).map(([key, def]) => ({
      id: key,
      ...def,
    }));
  }

  getSubscription(userId) {
    const row = this.stmts.findByUserId.get(userId);
    if (!row) return null;
    return {
      id: row.id,
      userId: row.user_id,
      tier: row.tier,
      status: row.status,
      startedAt: row.started_at,
      expiresAt: row.expires_at,
      cancelledAt: row.cancelled_at,
      createdAt: row.created_at,
      updatedAt: row.updated_at,
      tierDetails: this.getTierDefinition(row.tier),
    };
  }

  createSubscription(userId, tier = 'free') {
    if (!this.tiers[tier]) {
      throw new ValidationError(`Invalid tier: ${tier}`);
    }
    const existing = this.stmts.findByUserId.get(userId);
    if (existing) {
      return this.changeTier(userId, tier);
    }

    const id = generateId('sub');
    const now = nowIso();
    const expiresAt = tier === 'free' ? null : new Date(Date.now() + 30 * 24 * 60 * 60 * 1000).toISOString();

    this.stmts.insert.run(id, userId, tier, now, expiresAt, now, now);
    return this.getSubscription(userId);
  }

  changeTier(userId, newTier) {
    if (!this.tiers[newTier]) {
      throw new ValidationError(`Invalid tier: ${newTier}`);
    }
    const existing = this.stmts.findByUserId.get(userId);
    if (!existing) {
      return this.createSubscription(userId, newTier);
    }

    this.stmts.updateTier.run(newTier, nowIso(), userId);

    const userUpdateStmt = this.db.prepare('UPDATE users SET tier = ?, updated_at = ? WHERE id = ?');
    userUpdateStmt.run(newTier, nowIso(), userId);

    return this.getSubscription(userId);
  }

  cancelSubscription(userId) {
    const existing = this.stmts.findByUserId.get(userId);
    if (!existing) throw new NotFoundError('Subscription');

    const now = nowIso();
    this.stmts.cancel.run(now, now, userId);
    return this.getSubscription(userId);
  }

  /**
   * Check if a user can create a new job based on their tier limits.
   */
  checkJobQuota(userId, tier) {
    const def = this.getTierDefinition(tier);
    const today = new Date();
    today.setHours(0, 0, 0, 0);

    const count = this.stmts.countJobsToday.get(userId, today.toISOString()).count;
    if (count >= def.limits.jobsPerDay) {
      throw new QuotaExceededError(`Daily job limit (${def.limits.jobsPerDay} jobs)`);
    }
    return { used: count, limit: def.limits.jobsPerDay };
  }

  /**
   * Check if a requested target count is within the tier limit.
   */
  checkTargetCountLimit(tier, requestedCount) {
    const def = this.getTierDefinition(tier);
    if (requestedCount > def.limits.maxTargetCount) {
      throw new QuotaExceededError(
        `Target count limit (${def.limits.maxTargetCount} items for ${def.name} tier)`
      );
    }
    return true;
  }

  /**
   * Check if a provider is allowed for the user's tier.
   */
  checkProviderAccess(tier, provider) {
    const def = this.getTierDefinition(tier);
    if (!def.providers.includes(provider)) {
      throw new QuotaExceededError(
        `Provider '${provider}' is not available on ${def.name} tier`
      );
    }
    return true;
  }

  /**
   * Check if a feature is included in the user's tier.
   */
  hasFeature(tier, featureName) {
    const def = this.getTierDefinition(tier);
    return def.features.includes(featureName);
  }
}

module.exports = { SubscriptionService, TIER_DEFINITIONS };
