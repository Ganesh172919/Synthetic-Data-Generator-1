'use strict';

const { ValidationError } = require('./errors');

const EMAIL_RE = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
const USERNAME_RE = /^[a-zA-Z0-9_-]{3,40}$/;

const validateEmail = (email) => {
  if (typeof email !== 'string' || !EMAIL_RE.test(email.trim())) {
    throw new ValidationError('Invalid email address', { email: 'Must be a valid email' });
  }
  return email.trim().toLowerCase();
};

const validateUsername = (username) => {
  if (typeof username !== 'string' || !USERNAME_RE.test(username.trim())) {
    throw new ValidationError(
      'Invalid username. Must be 3-40 characters, alphanumeric, hyphens, or underscores.',
      { username: 'Must be 3-40 chars, alphanumeric/hyphens/underscores' }
    );
  }
  return username.trim();
};

const validatePassword = (password) => {
  if (typeof password !== 'string' || password.length < 8) {
    throw new ValidationError('Password must be at least 8 characters', {
      password: 'Minimum 8 characters',
    });
  }
  if (password.length > 128) {
    throw new ValidationError('Password must not exceed 128 characters', {
      password: 'Maximum 128 characters',
    });
  }
  return password;
};

const validateString = (value, fieldName, { minLength = 1, maxLength = 500 } = {}) => {
  if (typeof value !== 'string') {
    throw new ValidationError(`${fieldName} must be a string`, {
      [fieldName]: 'Must be a string',
    });
  }
  const trimmed = value.trim();
  if (trimmed.length < minLength) {
    throw new ValidationError(`${fieldName} must be at least ${minLength} characters`, {
      [fieldName]: `Minimum ${minLength} characters`,
    });
  }
  if (trimmed.length > maxLength) {
    throw new ValidationError(`${fieldName} must not exceed ${maxLength} characters`, {
      [fieldName]: `Maximum ${maxLength} characters`,
    });
  }
  return trimmed;
};

const validateInt = (value, fieldName, { min = 0, max = Number.MAX_SAFE_INTEGER } = {}) => {
  const parsed = Number.parseInt(value, 10);
  if (Number.isNaN(parsed)) {
    throw new ValidationError(`${fieldName} must be an integer`, {
      [fieldName]: 'Must be an integer',
    });
  }
  if (parsed < min || parsed > max) {
    throw new ValidationError(`${fieldName} must be between ${min} and ${max}`, {
      [fieldName]: `Must be between ${min} and ${max}`,
    });
  }
  return parsed;
};

const validateEnum = (value, fieldName, allowedValues) => {
  if (!allowedValues.includes(value)) {
    throw new ValidationError(`${fieldName} must be one of: ${allowedValues.join(', ')}`, {
      [fieldName]: `Must be one of: ${allowedValues.join(', ')}`,
    });
  }
  return value;
};

const sanitizeHtml = (str) => {
  if (typeof str !== 'string') return '';
  return str
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#x27;');
};

module.exports = {
  validateEmail,
  validateUsername,
  validatePassword,
  validateString,
  validateInt,
  validateEnum,
  sanitizeHtml,
};
