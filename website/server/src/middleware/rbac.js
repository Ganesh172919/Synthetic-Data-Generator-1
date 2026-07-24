'use strict';

const { AuthorizationError } = require('../utils/errors');

/**
 * Role-based access control middleware.
 * Checks if authenticated user has the required role.
 */
const requireRole = (...allowedRoles) => {
  return (req, res, next) => {
    if (!req.user) {
      return res.status(401).json({ error: 'Authentication required', code: 'AUTHENTICATION_ERROR' });
    }

    const userRole = req.user.role || 'user';
    if (!allowedRoles.includes(userRole)) {
      return res.status(403).json({
        error: 'Insufficient permissions',
        code: 'AUTHORIZATION_ERROR',
        required: allowedRoles,
        current: userRole,
      });
    }

    next();
  };
};

/**
 * Require the user to be authenticated (any role).
 */
const requireAuth = (req, res, next) => {
  if (!req.user) {
    return res.status(401).json({ error: 'Authentication required', code: 'AUTHENTICATION_ERROR' });
  }
  next();
};

/**
 * Require the user to be the owner of the resource or an admin.
 * Expects the resource owner's user ID to be available at req.params[paramName].
 */
const requireOwnerOrAdmin = (paramName = 'userId') => {
  return (req, res, next) => {
    if (!req.user) {
      return res.status(401).json({ error: 'Authentication required', code: 'AUTHENTICATION_ERROR' });
    }

    const resourceOwnerId = req.params[paramName];
    if (req.user.role === 'admin' || req.user.id === resourceOwnerId) {
      return next();
    }

    return res.status(403).json({
      error: 'You can only access your own resources',
      code: 'AUTHORIZATION_ERROR',
    });
  };
};

/**
 * Require specific API key scopes (for API key auth).
 */
const requireScopes = (...requiredScopes) => {
  return (req, res, next) => {
    if (!req.user) {
      return res.status(401).json({ error: 'Authentication required', code: 'AUTHENTICATION_ERROR' });
    }

    if (req.user.authMethod !== 'api_key') {
      return next();
    }

    const userScopes = req.user.scopes || [];
    const missing = requiredScopes.filter((s) => !userScopes.includes(s));

    if (missing.length > 0) {
      return res.status(403).json({
        error: `Missing required scopes: ${missing.join(', ')}`,
        code: 'INSUFFICIENT_SCOPES',
      });
    }

    next();
  };
};

module.exports = { requireRole, requireAuth, requireOwnerOrAdmin, requireScopes };
