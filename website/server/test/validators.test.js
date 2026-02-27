'use strict';

const { describe, it } = require('node:test');
const assert = require('node:assert/strict');

const {
  validateEmail,
  validateUsername,
  validatePassword,
  validateString,
  validateInt,
  validateEnum,
  sanitizeHtml,
} = require('../src/utils/validators');

describe('validators', () => {
  describe('validateEmail', () => {
    it('accepts valid email', () => {
      assert.equal(validateEmail('user@example.com'), 'user@example.com');
      assert.equal(validateEmail('  USER@EXAMPLE.COM  '), 'user@example.com');
    });

    it('rejects invalid email', () => {
      assert.throws(() => validateEmail('not-an-email'), /Invalid email/);
      assert.throws(() => validateEmail(''), /Invalid email/);
      assert.throws(() => validateEmail(123), /Invalid email/);
    });
  });

  describe('validateUsername', () => {
    it('accepts valid username', () => {
      assert.equal(validateUsername('alice'), 'alice');
      assert.equal(validateUsername('user-123_test'), 'user-123_test');
    });

    it('rejects short username', () => {
      assert.throws(() => validateUsername('ab'), /Invalid username/);
    });

    it('rejects invalid chars', () => {
      assert.throws(() => validateUsername('user name'), /Invalid username/);
      assert.throws(() => validateUsername('user@name'), /Invalid username/);
    });
  });

  describe('validatePassword', () => {
    it('accepts valid password', () => {
      assert.equal(validatePassword('12345678'), '12345678');
    });

    it('rejects short password', () => {
      assert.throws(() => validatePassword('1234567'), /at least 8/);
    });
  });

  describe('validateString', () => {
    it('trims and validates string', () => {
      assert.equal(validateString('  hello  ', 'test'), 'hello');
    });

    it('rejects empty string', () => {
      assert.throws(() => validateString('', 'test'), /at least 1/);
    });

    it('rejects string over maxLength', () => {
      const long = 'a'.repeat(600);
      assert.throws(() => validateString(long, 'test'), /must not exceed/);
    });
  });

  describe('validateInt', () => {
    it('parses and validates integer', () => {
      assert.equal(validateInt('42', 'count'), 42);
      assert.equal(validateInt(7, 'count'), 7);
    });

    it('rejects non-integer', () => {
      assert.throws(() => validateInt('abc', 'count'), /must be an integer/);
    });

    it('rejects out of range', () => {
      assert.throws(() => validateInt(5, 'count', { min: 10 }), /must be between/);
    });
  });

  describe('validateEnum', () => {
    it('accepts valid value', () => {
      assert.equal(validateEnum('b', 'test', ['a', 'b', 'c']), 'b');
    });

    it('rejects invalid value', () => {
      assert.throws(() => validateEnum('d', 'test', ['a', 'b']), /must be one of/);
    });
  });

  describe('sanitizeHtml', () => {
    it('escapes HTML entities', () => {
      assert.equal(sanitizeHtml('<script>alert("xss")</script>'), '&lt;script&gt;alert(&quot;xss&quot;)&lt;/script&gt;');
    });

    it('handles non-string input', () => {
      assert.equal(sanitizeHtml(null), '');
      assert.equal(sanitizeHtml(123), '');
    });
  });
});
