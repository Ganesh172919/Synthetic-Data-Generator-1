'use strict';

const { generateId } = require('../utils/crypto');
const { nowIso } = require('../db');

/**
 * Usage tracking service for metering and analytics.
 * Records resource consumption per user for billing and quota enforcement.
 */

const RESOURCE_TYPES = {
  JOB_CREATED: 'job_created',
  ROWS_GENERATED: 'rows_generated',
  API_CALL: 'api_call',
  DOWNLOAD: 'download',
  DOMAIN_CREATED: 'domain_created',
  TOKEN_USED: 'token_used',
};

class UsageService {
  constructor(db) {
    this.db = db;

    this.stmts = {
      insert: db.prepare(`
        INSERT INTO usage_records (user_id, resource_type, quantity, metadata_json, recorded_at)
        VALUES (?, ?, ?, ?, ?)
      `),
      sumByUserAndType: db.prepare(`
        SELECT COALESCE(SUM(quantity), 0) AS total
        FROM usage_records
        WHERE user_id = ? AND resource_type = ? AND recorded_at >= ?
      `),
      dailyByUser: db.prepare(`
        SELECT resource_type, SUM(quantity) AS total, COUNT(*) AS count
        FROM usage_records
        WHERE user_id = ? AND recorded_at >= ?
        GROUP BY resource_type
      `),
      aggregateByType: db.prepare(`
        SELECT resource_type, SUM(quantity) AS total, COUNT(*) AS events
        FROM usage_records
        WHERE recorded_at >= ?
        GROUP BY resource_type
      `),
      topUsers: db.prepare(`
        SELECT user_id, SUM(quantity) AS total
        FROM usage_records
        WHERE resource_type = ? AND recorded_at >= ?
        GROUP BY user_id
        ORDER BY total DESC
        LIMIT ?
      `),
      userTimeline: db.prepare(`
        SELECT
          DATE(recorded_at) AS date,
          resource_type,
          SUM(quantity) AS total
        FROM usage_records
        WHERE user_id = ? AND recorded_at >= ?
        GROUP BY DATE(recorded_at), resource_type
        ORDER BY DATE(recorded_at)
      `),
      cleanOld: db.prepare('DELETE FROM usage_records WHERE recorded_at < ?'),
    };
  }

  record(userId, resourceType, quantity = 1, metadata = null) {
    const metaJson = metadata ? JSON.stringify(metadata) : null;
    this.stmts.insert.run(userId, resourceType, quantity, metaJson, nowIso());
  }

  /**
   * Get total usage of a resource type for a user since a given date.
   */
  getUsage(userId, resourceType, sinceIso) {
    return this.stmts.sumByUserAndType.get(userId, resourceType, sinceIso).total;
  }

  /**
   * Get today's usage breakdown for a user.
   */
  getTodayUsage(userId) {
    const today = new Date();
    today.setHours(0, 0, 0, 0);
    const rows = this.stmts.dailyByUser.all(userId, today.toISOString());
    return rows.reduce((acc, row) => {
      acc[row.resource_type] = { total: row.total, events: row.count };
      return acc;
    }, {});
  }

  /**
   * Get aggregate usage metrics for the whole system.
   */
  getSystemUsage(sinceIso) {
    const rows = this.stmts.aggregateByType.all(sinceIso);
    return rows.reduce((acc, row) => {
      acc[row.resource_type] = { total: row.total, events: row.events };
      return acc;
    }, {});
  }

  /**
   * Get top users by a specific resource type.
   */
  getTopUsers(resourceType, sinceIso, limit = 10) {
    return this.stmts.topUsers.all(resourceType, sinceIso, limit);
  }

  /**
   * Get usage timeline for a user (daily breakdown).
   */
  getUserTimeline(userId, sinceIso) {
    return this.stmts.userTimeline.all(userId, sinceIso);
  }

  /**
   * Clean up old usage records beyond retention period.
   */
  cleanup(retentionDays = 90) {
    const threshold = new Date(Date.now() - retentionDays * 24 * 60 * 60 * 1000).toISOString();
    const result = this.stmts.cleanOld.run(threshold);
    return result.changes;
  }
}

module.exports = { UsageService, RESOURCE_TYPES };
