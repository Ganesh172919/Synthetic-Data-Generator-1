'use strict';

/**
 * Feature gate middleware.
 * Checks if the user's tier has access to a specific feature.
 */

const createFeatureGate = (featureFlagService) => {
  return (featureName) => {
    return (req, res, next) => {
      const tier = req.user?.tier || 'free';
      const role = req.user?.role || 'user';

      try {
        featureFlagService.checkAccess(featureName, tier, role);
        next();
      } catch (err) {
        res.status(err.statusCode || 403).json(err.toJSON ? err.toJSON() : { error: err.message });
      }
    };
  };
};

/**
 * Tier requirement middleware - simpler check without feature flags DB lookup.
 */
const requireTier = (...allowedTiers) => {
  return (req, res, next) => {
    const userTier = req.user?.tier || 'free';
    if (!allowedTiers.includes(userTier)) {
      return res.status(403).json({
        error: `This feature requires a ${allowedTiers.join(' or ')} plan`,
        code: 'TIER_REQUIRED',
        requiredTier: allowedTiers,
        currentTier: userTier,
      });
    }
    next();
  };
};

module.exports = { createFeatureGate, requireTier };
