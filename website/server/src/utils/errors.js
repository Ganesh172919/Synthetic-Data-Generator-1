'use strict';

/**
 * Custom error classes for structured API error handling.
 * Each error type maps to a specific HTTP status code and
 * machine-readable error code for client consumption.
 */

class AppError extends Error {
  constructor(message, statusCode = 500, code = 'INTERNAL_ERROR') {
    super(message);
    this.name = this.constructor.name;
    this.statusCode = statusCode;
    this.code = code;
    Error.captureStackTrace(this, this.constructor);
  }

  toJSON() {
    return {
      error: this.message,
      code: this.code,
      statusCode: this.statusCode,
    };
  }
}

class ValidationError extends AppError {
  constructor(message, fields = null) {
    super(message, 400, 'VALIDATION_ERROR');
    this.fields = fields;
  }

  toJSON() {
    const json = super.toJSON();
    if (this.fields) {
      json.fields = this.fields;
    }
    return json;
  }
}

class AuthenticationError extends AppError {
  constructor(message = 'Authentication required') {
    super(message, 401, 'AUTHENTICATION_ERROR');
  }
}

class AuthorizationError extends AppError {
  constructor(message = 'Insufficient permissions') {
    super(message, 403, 'AUTHORIZATION_ERROR');
  }
}

class NotFoundError extends AppError {
  constructor(resource = 'Resource') {
    super(`${resource} not found`, 404, 'NOT_FOUND');
  }
}

class ConflictError extends AppError {
  constructor(message = 'Resource already exists') {
    super(message, 409, 'CONFLICT');
  }
}

class RateLimitError extends AppError {
  constructor(message = 'Rate limit exceeded', retryAfterSeconds = null) {
    super(message, 429, 'RATE_LIMIT_EXCEEDED');
    this.retryAfter = retryAfterSeconds;
  }
}

class QuotaExceededError extends AppError {
  constructor(resource = 'quota') {
    super(`${resource} exceeded. Upgrade your plan for higher limits.`, 402, 'QUOTA_EXCEEDED');
  }
}

class FeatureDisabledError extends AppError {
  constructor(feature = 'Feature') {
    super(`${feature} is not available on your current plan`, 403, 'FEATURE_DISABLED');
  }
}

const errorHandler = (err, req, res, _next) => {
  if (err instanceof AppError) {
    const body = err.toJSON();
    if (err.retryAfter) {
      res.setHeader('Retry-After', err.retryAfter);
    }
    res.status(err.statusCode).json(body);
    return;
  }

  if (err.type === 'entity.parse.failed') {
    res.status(400).json({ error: 'Invalid JSON in request body', code: 'PARSE_ERROR' });
    return;
  }

  const logger = req.log || console;
  logger.error({ err }, 'Unhandled server error');
  res.status(500).json({ error: 'Internal server error', code: 'INTERNAL_ERROR' });
};

module.exports = {
  AppError,
  ValidationError,
  AuthenticationError,
  AuthorizationError,
  NotFoundError,
  ConflictError,
  RateLimitError,
  QuotaExceededError,
  FeatureDisabledError,
  errorHandler,
};
