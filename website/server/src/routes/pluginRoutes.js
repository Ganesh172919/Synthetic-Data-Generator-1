'use strict';

const express = require('express');
const { requireAuth } = require('../middleware/rbac');

/**
 * Plugin routes (user-facing).
 * Admins manage plugins via /api/admin/plugins.
 * Users can browse enabled plugins here.
 */
const createPluginRoutes = ({ pluginService, featureFlagService }) => {
  const router = express.Router();

  // List enabled plugins (public)
  router.get('/', (_req, res, next) => {
    try {
      const plugins = pluginService.listEnabled();
      res.json({ plugins });
    } catch (err) {
      next(err);
    }
  });

  // Get plugin details
  router.get('/:id', (req, res, next) => {
    try {
      const plugin = pluginService.getById(req.params.id);
      res.json(plugin);
    } catch (err) {
      next(err);
    }
  });

  // Get available plugin hooks
  router.get('/meta/hooks', (_req, res) => {
    res.json({ hooks: pluginService.getAvailableHooks() });
  });

  return router;
};

module.exports = { createPluginRoutes };
