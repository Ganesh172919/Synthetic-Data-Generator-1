'use strict';

const crypto = require('crypto');

const HASH_ITERATIONS = 100000;
const HASH_KEYLEN = 64;
const HASH_DIGEST = 'sha512';
const SALT_BYTES = 32;

/**
 * Hash a password with PBKDF2 + random salt.
 * Returns "salt:hash" hex string.
 */
const hashPassword = (password) => {
  const salt = crypto.randomBytes(SALT_BYTES).toString('hex');
  const hash = crypto.pbkdf2Sync(password, salt, HASH_ITERATIONS, HASH_KEYLEN, HASH_DIGEST).toString('hex');
  return `${salt}:${hash}`;
};

/**
 * Verify a password against a "salt:hash" string.
 */
const verifyPassword = (password, storedHash) => {
  const [salt, hash] = storedHash.split(':');
  if (!salt || !hash) {
    return false;
  }
  const derived = crypto.pbkdf2Sync(password, salt, HASH_ITERATIONS, HASH_KEYLEN, HASH_DIGEST).toString('hex');
  return crypto.timingSafeEqual(Buffer.from(hash, 'hex'), Buffer.from(derived, 'hex'));
};

/**
 * Generate a secure random API key with prefix.
 * Format: "sg_live_<32 hex chars>"
 */
const generateApiKey = (prefix = 'sg_live') => {
  const random = crypto.randomBytes(24).toString('hex');
  return `${prefix}_${random}`;
};

/**
 * Hash an API key for storage (SHA-256).
 * Only the hash is stored; the raw key is shown once at creation.
 */
const hashApiKey = (key) => {
  return crypto.createHash('sha256').update(key).digest('hex');
};

/**
 * Minimal JWT implementation using HMAC-SHA256.
 * No external dependency required.
 */
const base64UrlEncode = (data) => {
  return Buffer.from(data).toString('base64url');
};

const base64UrlDecode = (str) => {
  return Buffer.from(str, 'base64url').toString('utf-8');
};

const signJwt = (payload, secret, expiresInSeconds = 86400) => {
  const header = { alg: 'HS256', typ: 'JWT' };
  const now = Math.floor(Date.now() / 1000);

  const fullPayload = {
    ...payload,
    iat: now,
    exp: now + expiresInSeconds,
  };

  const headerB64 = base64UrlEncode(JSON.stringify(header));
  const payloadB64 = base64UrlEncode(JSON.stringify(fullPayload));
  const signature = crypto
    .createHmac('sha256', secret)
    .update(`${headerB64}.${payloadB64}`)
    .digest('base64url');

  return `${headerB64}.${payloadB64}.${signature}`;
};

const verifyJwt = (token, secret) => {
  const parts = token.split('.');
  if (parts.length !== 3) {
    return null;
  }

  const [headerB64, payloadB64, signature] = parts;

  const expectedSig = crypto
    .createHmac('sha256', secret)
    .update(`${headerB64}.${payloadB64}`)
    .digest('base64url');

  if (!crypto.timingSafeEqual(Buffer.from(signature), Buffer.from(expectedSig))) {
    return null;
  }

  try {
    const payload = JSON.parse(base64UrlDecode(payloadB64));
    const now = Math.floor(Date.now() / 1000);
    if (payload.exp && payload.exp < now) {
      return null;
    }
    return payload;
  } catch {
    return null;
  }
};

const generateId = (prefix = '') => {
  const uuid = crypto.randomUUID().replace(/-/g, '').slice(0, 12);
  return prefix ? `${prefix}_${uuid}` : uuid;
};

module.exports = {
  hashPassword,
  verifyPassword,
  generateApiKey,
  hashApiKey,
  signJwt,
  verifyJwt,
  generateId,
};
