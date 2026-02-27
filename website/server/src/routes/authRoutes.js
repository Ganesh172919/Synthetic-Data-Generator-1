'use strict';

const express = require('express');
const { signJwt } = require('../utils/crypto');
const { ValidationError } = require('../utils/errors');

/**
 * Authentication routes: register, login, profile, password change.
 */
const createAuthRoutes = ({ userService, subscriptionService, jwtSecret, jwtExpiresIn = 86400 }) => {
  const router = express.Router();

  router.post('/register', (req, res, next) => {
    try {
      const { email, username, password, displayName } = req.body || {};
      if (!email || !username || !password) {
        throw new ValidationError('Email, username, and password are required');
      }

      const user = userService.register({ email, username, password, displayName });
      subscriptionService.createSubscription(user.id, 'free');

      const token = signJwt(
        { sub: user.id, email: user.email, role: user.role, tier: user.tier },
        jwtSecret,
        jwtExpiresIn
      );

      res.status(201).json({
        user,
        token,
        expiresIn: jwtExpiresIn,
      });
    } catch (err) {
      next(err);
    }
  });

  router.post('/login', (req, res, next) => {
    try {
      const { email, password } = req.body || {};
      if (!email || !password) {
        throw new ValidationError('Email and password are required');
      }

      const user = userService.login(email, password);
      const token = signJwt(
        { sub: user.id, email: user.email, role: user.role, tier: user.tier },
        jwtSecret,
        jwtExpiresIn
      );

      res.json({
        user,
        token,
        expiresIn: jwtExpiresIn,
      });
    } catch (err) {
      next(err);
    }
  });

  router.get('/profile', (req, res, next) => {
    try {
      if (!req.user) {
        return res.status(401).json({ error: 'Authentication required' });
      }
      const user = userService.getById(req.user.id);
      const subscription = subscriptionService.getSubscription(req.user.id);
      res.json({ user, subscription });
    } catch (err) {
      next(err);
    }
  });

  router.put('/profile', (req, res, next) => {
    try {
      if (!req.user) {
        return res.status(401).json({ error: 'Authentication required' });
      }
      const updated = userService.updateProfile(req.user.id, req.body || {});
      res.json({ user: updated });
    } catch (err) {
      next(err);
    }
  });

  router.post('/change-password', (req, res, next) => {
    try {
      if (!req.user) {
        return res.status(401).json({ error: 'Authentication required' });
      }
      const { currentPassword, newPassword } = req.body || {};
      if (!currentPassword || !newPassword) {
        throw new ValidationError('Current password and new password are required');
      }
      userService.changePassword(req.user.id, currentPassword, newPassword);
      res.json({ message: 'Password changed successfully' });
    } catch (err) {
      next(err);
    }
  });

  return router;
};

module.exports = { createAuthRoutes };
