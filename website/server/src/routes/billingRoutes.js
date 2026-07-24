'use strict';

const express = require('express');
const { requireAuth } = require('../middleware/rbac');

/**
 * Billing and subscription routes.
 */
const createBillingRoutes = ({ subscriptionService, usageService, analyticsService }) => {
  const router = express.Router();

  router.use(requireAuth);

  // Get available subscription tiers
  router.get('/tiers', (_req, res) => {
    res.json({ tiers: subscriptionService.getAllTiers() });
  });

  // Get current user's subscription
  router.get('/subscription', (req, res, next) => {
    try {
      const subscription = subscriptionService.getSubscription(req.user.id);
      if (!subscription) {
        return res.json({
          subscription: null,
          tierDetails: subscriptionService.getTierDefinition('free'),
        });
      }
      res.json({ subscription });
    } catch (err) {
      next(err);
    }
  });

  // Upgrade/downgrade subscription tier
  router.post('/subscription/change', (req, res, next) => {
    try {
      const { tier } = req.body || {};
      const subscription = subscriptionService.changeTier(req.user.id, tier);
      analyticsService.recordAudit(
        req.user.id, 'change_subscription', 'subscription', subscription.id,
        { oldTier: req.user.tier, newTier: tier }, req.ip
      );
      res.json({ subscription, message: `Subscription changed to ${tier}` });
    } catch (err) {
      next(err);
    }
  });

  // Cancel subscription
  router.post('/subscription/cancel', (req, res, next) => {
    try {
      const subscription = subscriptionService.cancelSubscription(req.user.id);
      analyticsService.recordAudit(
        req.user.id, 'cancel_subscription', 'subscription', subscription.id,
        null, req.ip
      );
      res.json({ subscription, message: 'Subscription cancelled' });
    } catch (err) {
      next(err);
    }
  });

  // Get current usage summary
  router.get('/usage', (req, res, next) => {
    try {
      const today = usageService.getTodayUsage(req.user.id);
      const tier = req.user.tier || 'free';
      const limits = subscriptionService.getTierDefinition(tier).limits;

      res.json({
        today,
        limits,
        tier,
      });
    } catch (err) {
      next(err);
    }
  });

  // Get usage timeline (daily breakdown)
  router.get('/usage/timeline', (req, res, next) => {
    try {
      const days = Math.min(90, Math.max(1, parseInt(req.query.days || '30', 10)));
      const since = new Date(Date.now() - days * 24 * 60 * 60 * 1000).toISOString();
      const timeline = usageService.getUserTimeline(req.user.id, since);
      res.json({ timeline, days });
    } catch (err) {
      next(err);
    }
  });

  // Check quota status before creating a job
  router.get('/quota-check', (req, res, next) => {
    try {
      const tier = req.user.tier || 'free';
      const quota = subscriptionService.checkJobQuota(req.user.id, tier);
      const limits = subscriptionService.getTierDefinition(tier).limits;

      res.json({
        canCreate: true,
        jobsUsed: quota.used,
        jobsLimit: quota.limit,
        limits,
        tier,
      });
    } catch (err) {
      if (err.code === 'QUOTA_EXCEEDED') {
        return res.json({
          canCreate: false,
          error: err.message,
          tier: req.user.tier || 'free',
        });
      }
      next(err);
    }
  });

  return router;
};

module.exports = { createBillingRoutes };
