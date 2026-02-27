'use strict';

const express = require('express');
const { requireRole, requireAuth } = require('../middleware/rbac');

/**
 * Admin routes for system management.
 * All routes require admin role.
 */
const createAdminRoutes = ({
  userService,
  subscriptionService,
  analyticsService,
  featureFlagService,
  pluginService,
  usageService,
}) => {
  const router = express.Router();

  router.use(requireAuth);
  router.use(requireRole('admin'));

  // --- User management ---

  router.get('/users', (req, res, next) => {
    try {
      const page = Math.max(1, parseInt(req.query.page || '1', 10));
      const limit = Math.min(100, Math.max(1, parseInt(req.query.limit || '20', 10)));
      const result = userService.list({ page, limit });
      res.json(result);
    } catch (err) {
      next(err);
    }
  });

  router.get('/users/:userId', (req, res, next) => {
    try {
      const user = userService.getById(req.params.userId);
      const subscription = subscriptionService.getSubscription(req.params.userId);
      res.json({ user, subscription });
    } catch (err) {
      next(err);
    }
  });

  router.put('/users/:userId/tier', (req, res, next) => {
    try {
      const { tier } = req.body || {};
      const user = userService.updateTier(req.params.userId, tier);
      subscriptionService.changeTier(req.params.userId, tier);
      analyticsService.recordAudit(
        req.user.id, 'change_tier', 'user', req.params.userId,
        { tier }, req.ip
      );
      res.json({ user });
    } catch (err) {
      next(err);
    }
  });

  router.put('/users/:userId/role', (req, res, next) => {
    try {
      const { role } = req.body || {};
      const user = userService.updateRole(req.params.userId, role);
      analyticsService.recordAudit(
        req.user.id, 'change_role', 'user', req.params.userId,
        { role }, req.ip
      );
      res.json({ user });
    } catch (err) {
      next(err);
    }
  });

  router.post('/users/:userId/deactivate', (req, res, next) => {
    try {
      userService.deactivate(req.params.userId);
      analyticsService.recordAudit(
        req.user.id, 'deactivate_user', 'user', req.params.userId, null, req.ip
      );
      res.json({ message: 'User deactivated' });
    } catch (err) {
      next(err);
    }
  });

  router.post('/users/:userId/activate', (req, res, next) => {
    try {
      userService.activate(req.params.userId);
      analyticsService.recordAudit(
        req.user.id, 'activate_user', 'user', req.params.userId, null, req.ip
      );
      res.json({ message: 'User activated' });
    } catch (err) {
      next(err);
    }
  });

  // --- Analytics ---

  router.get('/analytics/overview', (_req, res, next) => {
    try {
      res.json(analyticsService.getSystemOverview());
    } catch (err) {
      next(err);
    }
  });

  router.get('/analytics/throughput', (_req, res, next) => {
    try {
      res.json(analyticsService.getThroughputMetrics());
    } catch (err) {
      next(err);
    }
  });

  router.get('/analytics/trends', (req, res, next) => {
    try {
      const days = Math.min(365, Math.max(1, parseInt(req.query.days || '30', 10)));
      res.json(analyticsService.getDailyTrends(days));
    } catch (err) {
      next(err);
    }
  });

  router.get('/analytics/revenue', (_req, res, next) => {
    try {
      res.json(analyticsService.getRevenueMetrics());
    } catch (err) {
      next(err);
    }
  });

  router.get('/analytics/errors', (req, res, next) => {
    try {
      const days = Math.min(90, Math.max(1, parseInt(req.query.days || '7', 10)));
      res.json(analyticsService.getErrorAnalytics(days));
    } catch (err) {
      next(err);
    }
  });

  router.get('/analytics/popular-domains', (req, res, next) => {
    try {
      const limit = Math.min(50, Math.max(1, parseInt(req.query.limit || '10', 10)));
      res.json(analyticsService.getPopularDomains(limit));
    } catch (err) {
      next(err);
    }
  });

  // --- Feature flags ---

  router.get('/feature-flags', (_req, res, next) => {
    try {
      res.json({ flags: featureFlagService.listAll() });
    } catch (err) {
      next(err);
    }
  });

  router.put('/feature-flags/:name', (req, res, next) => {
    try {
      const updated = featureFlagService.updateFlag(req.params.name, req.body || {});
      analyticsService.recordAudit(
        req.user.id, 'update_feature_flag', 'feature_flag', req.params.name,
        req.body, req.ip
      );
      res.json(updated);
    } catch (err) {
      next(err);
    }
  });

  router.post('/feature-flags/:name/toggle', (req, res, next) => {
    try {
      const { enabled } = req.body || {};
      const updated = featureFlagService.toggle(req.params.name, Boolean(enabled));
      analyticsService.recordAudit(
        req.user.id, 'toggle_feature_flag', 'feature_flag', req.params.name,
        { enabled: Boolean(enabled) }, req.ip
      );
      res.json(updated);
    } catch (err) {
      next(err);
    }
  });

  // --- Plugins ---

  router.get('/plugins', (_req, res, next) => {
    try {
      res.json({ plugins: pluginService.listAll() });
    } catch (err) {
      next(err);
    }
  });

  router.post('/plugins', (req, res, next) => {
    try {
      const plugin = pluginService.register(req.body || {});
      analyticsService.recordAudit(
        req.user.id, 'register_plugin', 'plugin', plugin.id,
        { name: plugin.name }, req.ip
      );
      res.status(201).json(plugin);
    } catch (err) {
      next(err);
    }
  });

  router.post('/plugins/:id/enable', (req, res, next) => {
    try {
      const plugin = pluginService.enable(req.params.id);
      res.json(plugin);
    } catch (err) {
      next(err);
    }
  });

  router.post('/plugins/:id/disable', (req, res, next) => {
    try {
      const plugin = pluginService.disable(req.params.id);
      res.json(plugin);
    } catch (err) {
      next(err);
    }
  });

  router.delete('/plugins/:id', (req, res, next) => {
    try {
      pluginService.uninstall(req.params.id);
      res.json({ message: 'Plugin uninstalled' });
    } catch (err) {
      next(err);
    }
  });

  // --- Audit log ---

  router.get('/audit-log', (req, res, next) => {
    try {
      const limit = Math.min(100, Math.max(1, parseInt(req.query.limit || '50', 10)));
      const offset = Math.max(0, parseInt(req.query.offset || '0', 10));
      const entries = analyticsService.getAuditLog({
        userId: req.query.userId,
        action: req.query.action,
        limit,
        offset,
      });
      res.json({ entries });
    } catch (err) {
      next(err);
    }
  });

  // --- System usage ---

  router.get('/usage/system', (req, res, next) => {
    try {
      const days = Math.min(90, Math.max(1, parseInt(req.query.days || '30', 10)));
      const since = new Date(Date.now() - days * 24 * 60 * 60 * 1000).toISOString();
      res.json(usageService.getSystemUsage(since));
    } catch (err) {
      next(err);
    }
  });

  // --- User stats ---

  router.get('/stats', (_req, res, next) => {
    try {
      res.json(userService.getStats());
    } catch (err) {
      next(err);
    }
  });

  return router;
};

module.exports = { createAdminRoutes };
