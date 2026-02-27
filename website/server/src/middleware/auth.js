'use strict';

const { verifyJwt } = require('../utils/crypto');
const { AuthenticationError } = require('../utils/errors');

/**
 * Authentication middleware supporting multiple strategies:
 * 1. JWT Bearer token (Authorization: Bearer <jwt>)
 * 2. Platform API key (x-api-key: sg_live_xxx)
 * 3. Legacy static API keys (from config)
 * 4. Local bypass for development
 */

const extractBearerToken = (req) => {
  const auth = req.get('authorization');
  if (auth && auth.toLowerCase().startsWith('bearer ')) {
    return auth.slice(7).trim();
  }
  return null;
};

const extractApiKeyHeader = (req) => {
  return req.get('x-api-key')?.trim() || null;
};

const isLocalRequest = (req) => {
  const raw = (req.ip || req.socket?.remoteAddress || '').replace('::ffff:', '');
  return raw === '127.0.0.1' || raw === '::1' || raw.startsWith('127.');
};

/**
 * Create JWT + API key authentication middleware.
 *
 * @param {object} options
 * @param {string} options.jwtSecret - Secret for JWT verification
 * @param {object} options.apiKeyService - ApiKeyService instance
 * @param {string[]} options.legacyApiKeys - Static API keys from config
 * @param {string} options.authMode - 'none' | 'api_key' | 'jwt'
 */
const createAuthMiddleware = ({ jwtSecret, apiKeyService, legacyApiKeys = [], authMode = 'none' }) => {
  return (req, res, next) => {
    if (!req.path.startsWith('/api')) {
      return next();
    }

    const publicPaths = [
      '/api/health',
      '/api/auth/register',
      '/api/auth/login',
      '/api/tiers',
    ];
    if (publicPaths.some((p) => req.path === p || req.path.startsWith(p + '/'))) {
      return next();
    }

    if (authMode === 'none') {
      req.user = null;
      return next();
    }

    const bearerToken = extractBearerToken(req);
    if (bearerToken && jwtSecret) {
      const payload = verifyJwt(bearerToken, jwtSecret);
      if (payload) {
        req.user = {
          id: payload.sub,
          email: payload.email,
          role: payload.role || 'user',
          tier: payload.tier || 'free',
          authMethod: 'jwt',
        };
        return next();
      }
    }

    const apiKey = extractApiKeyHeader(req) || bearerToken;
    if (apiKey && apiKeyService) {
      const keyInfo = apiKeyService.validateKey(apiKey);
      if (keyInfo) {
        req.user = {
          id: keyInfo.userId,
          scopes: keyInfo.scopes,
          authMethod: 'api_key',
          keyId: keyInfo.keyId,
        };

        try {
          const userRow = apiKeyService.db.prepare('SELECT role, tier FROM users WHERE id = ?').get(keyInfo.userId);
          if (userRow) {
            req.user.role = userRow.role;
            req.user.tier = userRow.tier;
          }
        } catch {
          req.user.role = 'user';
          req.user.tier = 'free';
        }

        return next();
      }
    }

    if (apiKey && legacyApiKeys.includes(apiKey)) {
      req.user = { id: 'legacy', role: 'admin', tier: 'enterprise', authMethod: 'legacy_key' };
      return next();
    }

    if (isLocalRequest(req)) {
      req.user = null;
      return next();
    }

    if (authMode === 'jwt' || authMode === 'api_key') {
      return res.status(401).json({ error: 'Authentication required', code: 'AUTHENTICATION_ERROR' });
    }

    req.user = null;
    next();
  };
};

/**
 * Optional auth middleware - attaches user if token present, but doesn't reject.
 */
const createOptionalAuth = ({ jwtSecret, apiKeyService }) => {
  return (req, _res, next) => {
    const bearerToken = extractBearerToken(req);
    if (bearerToken && jwtSecret) {
      const payload = verifyJwt(bearerToken, jwtSecret);
      if (payload) {
        req.user = {
          id: payload.sub,
          email: payload.email,
          role: payload.role || 'user',
          tier: payload.tier || 'free',
          authMethod: 'jwt',
        };
        return next();
      }
    }

    const apiKey = extractApiKeyHeader(req);
    if (apiKey && apiKeyService) {
      const keyInfo = apiKeyService.validateKey(apiKey);
      if (keyInfo) {
        req.user = { id: keyInfo.userId, scopes: keyInfo.scopes, authMethod: 'api_key' };
        return next();
      }
    }

    req.user = null;
    next();
  };
};

module.exports = { createAuthMiddleware, createOptionalAuth };
