'use strict';

const { nowIso } = require('../db');
const { cache } = require('./cacheService');

/**
 * Analytics engine for system health, usage patterns, revenue metrics,
 * and intelligent recommendations.
 */

class AnalyticsService {
  constructor(db) {
    this.db = db;
  }

  /**
   * Get comprehensive system overview metrics.
   */
  getSystemOverview() {
    const cacheKey = 'analytics:system_overview';
    const cached = cache.get(cacheKey);
    if (cached) return cached;

    const jobStatusCounts = this.db
      .prepare('SELECT status, COUNT(*) AS count FROM jobs GROUP BY status')
      .all()
      .reduce((acc, row) => { acc[row.status] = row.count; return acc; }, {
        queued: 0, running: 0, completed: 0, failed: 0, stopped: 0,
      });

    const totalJobs = Object.values(jobStatusCounts).reduce((a, b) => a + b, 0);
    const successRate = totalJobs > 0
      ? ((jobStatusCounts.completed / totalJobs) * 100).toFixed(2)
      : '0.00';

    const totalRowsGenerated = this.db
      .prepare("SELECT COALESCE(SUM(generated_count), 0) AS total FROM jobs WHERE status = 'completed'")
      .get().total;

    const userCounts = this._safeQuery(
      'SELECT COUNT(*) AS total FROM users',
      {},
      { total: 0 }
    );

    const tierBreakdown = this._safeQuery(
      'SELECT tier, COUNT(*) AS count FROM users GROUP BY tier',
      null,
      []
    );

    const result = {
      jobs: { ...jobStatusCounts, total: totalJobs, successRate: `${successRate}%` },
      generation: { totalRowsGenerated },
      users: {
        total: userCounts.total || 0,
        byTier: Array.isArray(tierBreakdown)
          ? tierBreakdown.reduce((acc, r) => { acc[r.tier] = r.count; return acc; }, {})
          : {},
      },
      timestamp: nowIso(),
    };

    cache.set(cacheKey, result, 15000);
    return result;
  }

  /**
   * Get generation throughput metrics.
   */
  getThroughputMetrics() {
    const cacheKey = 'analytics:throughput';
    const cached = cache.get(cacheKey);
    if (cached) return cached;

    const recentJobs = this.db.prepare(`
      SELECT rate_items_per_sec, generated_count, target_count,
             started_at, completed_at, provider, domain
      FROM jobs
      WHERE status = 'completed' AND rate_items_per_sec > 0
      ORDER BY datetime(completed_at) DESC
      LIMIT 100
    `).all();

    const avgRate = recentJobs.length > 0
      ? recentJobs.reduce((sum, r) => sum + Number(r.rate_items_per_sec || 0), 0) / recentJobs.length
      : 0;

    const byProvider = {};
    const byDomain = {};
    for (const job of recentJobs) {
      const p = job.provider || 'unknown';
      if (!byProvider[p]) byProvider[p] = { count: 0, totalRate: 0 };
      byProvider[p].count++;
      byProvider[p].totalRate += Number(job.rate_items_per_sec || 0);

      const d = job.domain || 'unknown';
      if (!byDomain[d]) byDomain[d] = { count: 0, totalGenerated: 0 };
      byDomain[d].count++;
      byDomain[d].totalGenerated += Number(job.generated_count || 0);
    }

    for (const p of Object.keys(byProvider)) {
      byProvider[p].avgRate = (byProvider[p].totalRate / byProvider[p].count).toFixed(4);
    }

    const result = {
      sampleSize: recentJobs.length,
      avgItemsPerSec: Number(avgRate.toFixed(4)),
      byProvider,
      byDomain,
    };

    cache.set(cacheKey, result, 30000);
    return result;
  }

  /**
   * Get daily job volume trends (last N days).
   */
  getDailyTrends(days = 30) {
    const cacheKey = `analytics:daily_trends:${days}`;
    const cached = cache.get(cacheKey);
    if (cached) return cached;

    const since = new Date(Date.now() - days * 24 * 60 * 60 * 1000).toISOString();

    const result = this.db.prepare(`
      SELECT DATE(created_at) AS date,
             COUNT(*) AS jobs,
             SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) AS completed,
             SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) AS failed,
             SUM(generated_count) AS rows_generated
      FROM jobs
      WHERE created_at >= ?
      GROUP BY DATE(created_at)
      ORDER BY DATE(created_at)
    `).all(since);

    cache.set(cacheKey, result, 60000);
    return result;
  }

  /**
   * Get revenue-related metrics.
   */
  getRevenueMetrics() {
    const tierPricing = { free: 0, pro: 29, enterprise: 199 };

    const tierCounts = this._safeQuery(
      "SELECT tier, COUNT(*) AS count FROM users WHERE is_active = 1 GROUP BY tier",
      null,
      []
    );

    let mrr = 0;
    const breakdown = {};

    if (Array.isArray(tierCounts)) {
      for (const row of tierCounts) {
        const price = tierPricing[row.tier] || 0;
        const revenue = row.count * price;
        mrr += revenue;
        breakdown[row.tier] = { users: row.count, price, revenue };
      }
    }

    return {
      mrr,
      arr: mrr * 12,
      breakdown,
      timestamp: nowIso(),
    };
  }

  /**
   * Get most popular templates/domains.
   */
  getPopularDomains(limit = 10) {
    return this.db.prepare(`
      SELECT domain, COUNT(*) AS jobs, SUM(generated_count) AS total_rows
      FROM jobs
      GROUP BY domain
      ORDER BY jobs DESC
      LIMIT ?
    `).all(limit);
  }

  /**
   * Get error analytics for troubleshooting.
   */
  getErrorAnalytics(days = 7) {
    const since = new Date(Date.now() - days * 24 * 60 * 60 * 1000).toISOString();

    return this.db.prepare(`
      SELECT error_message, COUNT(*) AS occurrences,
             MAX(updated_at) AS last_seen
      FROM jobs
      WHERE status = 'failed' AND error_message IS NOT NULL
            AND created_at >= ?
      GROUP BY error_message
      ORDER BY occurrences DESC
      LIMIT 20
    `).all(since);
  }

  /**
   * Audit log recording for security-sensitive actions.
   */
  recordAudit(userId, action, resourceType, resourceId, details = null, ipAddress = null) {
    this.db.prepare(`
      INSERT INTO audit_log (user_id, action, resource_type, resource_id, details_json, ip_address, recorded_at)
      VALUES (?, ?, ?, ?, ?, ?, ?)
    `).run(userId, action, resourceType, resourceId, details ? JSON.stringify(details) : null, ipAddress, nowIso());
  }

  /**
   * Get recent audit log entries.
   */
  getAuditLog({ userId, action, limit = 50, offset = 0 } = {}) {
    if (userId && action) {
      return this.db.prepare(
        'SELECT * FROM audit_log WHERE user_id = ? AND action = ? ORDER BY id DESC LIMIT ? OFFSET ?'
      ).all(userId, action, limit, offset);
    }
    if (userId) {
      return this.db.prepare(
        'SELECT * FROM audit_log WHERE user_id = ? ORDER BY id DESC LIMIT ? OFFSET ?'
      ).all(userId, limit, offset);
    }
    return this.db.prepare(
      'SELECT * FROM audit_log ORDER BY id DESC LIMIT ? OFFSET ?'
    ).all(limit, offset);
  }

  /**
   * Safely execute a query, returning a fallback on error.
   */
  _safeQuery(sql, params, fallback) {
    try {
      if (params === null) {
        return this.db.prepare(sql).all();
      }
      return this.db.prepare(sql).get(params) || fallback;
    } catch {
      return fallback;
    }
  }
}

module.exports = { AnalyticsService };
