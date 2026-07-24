'use strict';

/**
 * Usage tracking middleware.
 * Automatically records API usage per user for metering and analytics.
 */

const { RESOURCE_TYPES } = require('../services/usageService');

const createUsageTracker = (usageService) => {
  return (req, res, next) => {
    const userId = req.user?.id;
    if (!userId || userId === 'legacy') {
      return next();
    }

    res.on('finish', () => {
      if (res.statusCode >= 200 && res.statusCode < 400) {
        try {
          usageService.record(userId, RESOURCE_TYPES.API_CALL, 1, {
            method: req.method,
            path: req.path,
            statusCode: res.statusCode,
          });
        } catch {
          // Don't fail requests due to usage tracking errors
        }
      }
    });

    next();
  };
};

/**
 * Track job creation usage.
 */
const createJobCreationTracker = (usageService) => {
  return (req, res, next) => {
    const userId = req.user?.id;

    res.on('finish', () => {
      if (res.statusCode === 200 && userId && userId !== 'legacy') {
        try {
          usageService.record(userId, RESOURCE_TYPES.JOB_CREATED, 1, {
            domain: req.body?.domain,
            targetCount: req.body?.targetCount,
            provider: req.body?.provider,
          });
        } catch {
          // Silent fail
        }
      }
    });

    next();
  };
};

/**
 * Track download usage.
 */
const createDownloadTracker = (usageService) => {
  return (req, res, next) => {
    const userId = req.user?.id;

    res.on('finish', () => {
      if (res.statusCode === 200 && userId && userId !== 'legacy') {
        try {
          usageService.record(userId, RESOURCE_TYPES.DOWNLOAD, 1, {
            jobId: req.params.jobId,
            format: req.params.format,
          });
        } catch {
          // Silent fail
        }
      }
    });

    next();
  };
};

module.exports = { createUsageTracker, createJobCreationTracker, createDownloadTracker };
