'use strict';

const { describe, it } = require('node:test');
const assert = require('node:assert/strict');

const {
  AppError,
  ValidationError,
  AuthenticationError,
  AuthorizationError,
  NotFoundError,
  ConflictError,
  RateLimitError,
  QuotaExceededError,
  FeatureDisabledError,
} = require('../src/utils/errors');

describe('error classes', () => {
  it('AppError has correct defaults', () => {
    const err = new AppError('something broke');
    assert.equal(err.message, 'something broke');
    assert.equal(err.statusCode, 500);
    assert.equal(err.code, 'INTERNAL_ERROR');
    assert.ok(err instanceof Error);
  });

  it('ValidationError is 400', () => {
    const err = new ValidationError('bad input', { field: 'required' });
    assert.equal(err.statusCode, 400);
    assert.equal(err.code, 'VALIDATION_ERROR');
    const json = err.toJSON();
    assert.deepEqual(json.fields, { field: 'required' });
  });

  it('AuthenticationError is 401', () => {
    const err = new AuthenticationError();
    assert.equal(err.statusCode, 401);
    assert.equal(err.message, 'Authentication required');
  });

  it('AuthorizationError is 403', () => {
    const err = new AuthorizationError();
    assert.equal(err.statusCode, 403);
  });

  it('NotFoundError is 404', () => {
    const err = new NotFoundError('User');
    assert.equal(err.statusCode, 404);
    assert.equal(err.message, 'User not found');
  });

  it('ConflictError is 409', () => {
    const err = new ConflictError();
    assert.equal(err.statusCode, 409);
  });

  it('RateLimitError is 429 with retryAfter', () => {
    const err = new RateLimitError('too fast', 60);
    assert.equal(err.statusCode, 429);
    assert.equal(err.retryAfter, 60);
  });

  it('QuotaExceededError is 402', () => {
    const err = new QuotaExceededError('API calls');
    assert.equal(err.statusCode, 402);
    assert.ok(err.message.includes('Upgrade'));
  });

  it('FeatureDisabledError is 403', () => {
    const err = new FeatureDisabledError('bulk_export');
    assert.equal(err.statusCode, 403);
    assert.ok(err.message.includes('not available'));
  });

  it('toJSON serializes correctly', () => {
    const err = new AppError('test', 418, 'TEAPOT');
    const json = err.toJSON();
    assert.equal(json.error, 'test');
    assert.equal(json.code, 'TEAPOT');
    assert.equal(json.statusCode, 418);
  });
});
