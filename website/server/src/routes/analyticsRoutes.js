'use strict';

const express = require('express');
const { requireAuth } = require('../middleware/rbac');

/**
 * Analytics routes for users to view their own usage.
 */
const createAnalyticsRoutes = ({ analyticsService, usageService }) => {
  const router = express.Router();

  router.use(requireAuth);

  // User's own usage timeline
  router.get('/usage', (req, res, next) => {
    try {
      const days = Math.min(90, Math.max(1, parseInt(req.query.days || '30', 10)));
      const since = new Date(Date.now() - days * 24 * 60 * 60 * 1000).toISOString();
      const timeline = usageService.getUserTimeline(req.user.id, since);
      const today = usageService.getTodayUsage(req.user.id);
      res.json({ timeline, today, days });
    } catch (err) {
      next(err);
    }
  });

  // Popular domains (public stats)
  router.get('/popular-domains', (_req, res, next) => {
    try {
      const domains = analyticsService.getPopularDomains(10);
      res.json({ domains });
    } catch (err) {
      next(err);
    }
  });

  // System throughput (public stats)
  router.get('/throughput', (_req, res, next) => {
    try {
      const metrics = analyticsService.getThroughputMetrics();
      res.json(metrics);
    } catch (err) {
      next(err);
    }
  });

  return router;
};

module.exports = { createAnalyticsRoutes };
