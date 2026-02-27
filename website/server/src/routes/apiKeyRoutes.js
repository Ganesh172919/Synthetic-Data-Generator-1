'use strict';

const express = require('express');
const { requireAuth } = require('../middleware/rbac');

/**
 * API key management routes.
 */
const createApiKeyRoutes = ({ apiKeyService, subscriptionService, analyticsService }) => {
  const router = express.Router();

  router.use(requireAuth);

  // List user's API keys
  router.get('/', (req, res, next) => {
    try {
      const keys = apiKeyService.listByUser(req.user.id);
      res.json({ keys });
    } catch (err) {
      next(err);
    }
  });

  // Create a new API key
  router.post('/', (req, res, next) => {
    try {
      const tier = req.user.tier || 'free';
      const limits = subscriptionService.getTierDefinition(tier).limits;

      const result = apiKeyService.create(req.user.id, {
        name: req.body?.name,
        scopes: req.body?.scopes || ['read', 'write'],
        maxKeysAllowed: limits.apiKeysAllowed,
      });

      analyticsService.recordAudit(
        req.user.id, 'create_api_key', 'api_key', result.id,
        { name: req.body?.name }, req.ip
      );

      res.status(201).json({
        key: result,
        message: 'API key created. Save the raw key now — it will not be shown again.',
      });
    } catch (err) {
      next(err);
    }
  });

  // Revoke an API key
  router.post('/:keyId/revoke', (req, res, next) => {
    try {
      apiKeyService.revoke(req.params.keyId, req.user.id);
      analyticsService.recordAudit(
        req.user.id, 'revoke_api_key', 'api_key', req.params.keyId,
        null, req.ip
      );
      res.json({ message: 'API key revoked' });
    } catch (err) {
      next(err);
    }
  });

  // Delete an API key permanently
  router.delete('/:keyId', (req, res, next) => {
    try {
      apiKeyService.deleteKey(req.params.keyId, req.user.id);
      analyticsService.recordAudit(
        req.user.id, 'delete_api_key', 'api_key', req.params.keyId,
        null, req.ip
      );
      res.json({ message: 'API key deleted' });
    } catch (err) {
      next(err);
    }
  });

  return router;
};

module.exports = { createApiKeyRoutes };
