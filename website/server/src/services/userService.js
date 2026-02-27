'use strict';

const { hashPassword, verifyPassword, generateId } = require('../utils/crypto');
const { validateEmail, validateUsername, validatePassword, validateString } = require('../utils/validators');
const { ConflictError, NotFoundError, AuthenticationError, ValidationError } = require('../utils/errors');
const { nowIso } = require('../db');

class UserService {
  constructor(db) {
    this.db = db;

    this.stmts = {
      findById: db.prepare('SELECT * FROM users WHERE id = ?'),
      findByEmail: db.prepare('SELECT * FROM users WHERE email = ?'),
      findByUsername: db.prepare('SELECT * FROM users WHERE username = ?'),
      insert: db.prepare(`
        INSERT INTO users (id, email, username, password_hash, display_name, role, tier, is_active, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, 1, ?, ?)
      `),
      updateProfile: db.prepare(`
        UPDATE users SET display_name = ?, username = ?, updated_at = ? WHERE id = ?
      `),
      updatePassword: db.prepare('UPDATE users SET password_hash = ?, updated_at = ? WHERE id = ?'),
      updateTier: db.prepare('UPDATE users SET tier = ?, updated_at = ? WHERE id = ?'),
      updateRole: db.prepare('UPDATE users SET role = ?, updated_at = ? WHERE id = ?'),
      updateLastLogin: db.prepare('UPDATE users SET last_login_at = ?, updated_at = ? WHERE id = ?'),
      deactivate: db.prepare('UPDATE users SET is_active = 0, updated_at = ? WHERE id = ?'),
      activate: db.prepare('UPDATE users SET is_active = 1, updated_at = ? WHERE id = ?'),
      listUsers: db.prepare('SELECT * FROM users ORDER BY datetime(created_at) DESC LIMIT ? OFFSET ?'),
      countUsers: db.prepare('SELECT COUNT(*) AS count FROM users'),
      countByTier: db.prepare('SELECT tier, COUNT(*) AS count FROM users GROUP BY tier'),
      searchByEmail: db.prepare('SELECT * FROM users WHERE email LIKE ? LIMIT ?'),
    };
  }

  _sanitizeUser(row) {
    if (!row) return null;
    const { password_hash, ...user } = row;
    return {
      id: user.id,
      email: user.email,
      username: user.username,
      displayName: user.display_name,
      role: user.role,
      tier: user.tier,
      isActive: Boolean(user.is_active),
      emailVerified: Boolean(user.email_verified),
      lastLoginAt: user.last_login_at,
      createdAt: user.created_at,
      updatedAt: user.updated_at,
    };
  }

  register({ email, username, password, displayName = null, role = 'user', tier = 'free' }) {
    email = validateEmail(email);
    username = validateUsername(username);
    validatePassword(password);

    if (this.stmts.findByEmail.get(email)) {
      throw new ConflictError('An account with this email already exists');
    }
    if (this.stmts.findByUsername.get(username)) {
      throw new ConflictError('This username is already taken');
    }

    const id = generateId('usr');
    const now = nowIso();
    const passwordHash = hashPassword(password);
    const safeDisplayName = displayName ? validateString(displayName, 'displayName', { maxLength: 100 }) : username;

    this.stmts.insert.run(id, email, username, passwordHash, safeDisplayName, role, tier, now, now);

    return this._sanitizeUser(this.stmts.findById.get(id));
  }

  login(email, password) {
    email = validateEmail(email);
    validatePassword(password);

    const row = this.stmts.findByEmail.get(email);
    if (!row) {
      throw new AuthenticationError('Invalid email or password');
    }

    if (!row.is_active) {
      throw new AuthenticationError('Account is deactivated');
    }

    if (!verifyPassword(password, row.password_hash)) {
      throw new AuthenticationError('Invalid email or password');
    }

    const now = nowIso();
    this.stmts.updateLastLogin.run(now, now, row.id);

    return this._sanitizeUser(this.stmts.findById.get(row.id));
  }

  getById(userId) {
    const row = this.stmts.findById.get(userId);
    if (!row) throw new NotFoundError('User');
    return this._sanitizeUser(row);
  }

  getByEmail(email) {
    const row = this.stmts.findByEmail.get(email);
    return this._sanitizeUser(row);
  }

  updateProfile(userId, { displayName, username }) {
    const existing = this.stmts.findById.get(userId);
    if (!existing) throw new NotFoundError('User');

    const newUsername = username ? validateUsername(username) : existing.username;
    const newDisplayName = displayName
      ? validateString(displayName, 'displayName', { maxLength: 100 })
      : existing.display_name;

    if (newUsername !== existing.username) {
      const conflict = this.stmts.findByUsername.get(newUsername);
      if (conflict && conflict.id !== userId) {
        throw new ConflictError('This username is already taken');
      }
    }

    this.stmts.updateProfile.run(newDisplayName, newUsername, nowIso(), userId);
    return this._sanitizeUser(this.stmts.findById.get(userId));
  }

  changePassword(userId, currentPassword, newPassword) {
    const row = this.stmts.findById.get(userId);
    if (!row) throw new NotFoundError('User');

    validatePassword(currentPassword);
    validatePassword(newPassword);

    if (!verifyPassword(currentPassword, row.password_hash)) {
      throw new AuthenticationError('Current password is incorrect');
    }

    const newHash = hashPassword(newPassword);
    this.stmts.updatePassword.run(newHash, nowIso(), userId);
    return true;
  }

  updateTier(userId, tier) {
    const validTiers = ['free', 'pro', 'enterprise'];
    if (!validTiers.includes(tier)) {
      throw new ValidationError(`Invalid tier. Must be one of: ${validTiers.join(', ')}`);
    }
    const existing = this.stmts.findById.get(userId);
    if (!existing) throw new NotFoundError('User');

    this.stmts.updateTier.run(tier, nowIso(), userId);
    return this._sanitizeUser(this.stmts.findById.get(userId));
  }

  updateRole(userId, role) {
    const validRoles = ['user', 'admin'];
    if (!validRoles.includes(role)) {
      throw new ValidationError(`Invalid role. Must be one of: ${validRoles.join(', ')}`);
    }
    const existing = this.stmts.findById.get(userId);
    if (!existing) throw new NotFoundError('User');

    this.stmts.updateRole.run(role, nowIso(), userId);
    return this._sanitizeUser(this.stmts.findById.get(userId));
  }

  deactivate(userId) {
    const existing = this.stmts.findById.get(userId);
    if (!existing) throw new NotFoundError('User');
    this.stmts.deactivate.run(nowIso(), userId);
    return true;
  }

  activate(userId) {
    const existing = this.stmts.findById.get(userId);
    if (!existing) throw new NotFoundError('User');
    this.stmts.activate.run(nowIso(), userId);
    return true;
  }

  list({ page = 1, limit = 20 } = {}) {
    const offset = (Math.max(1, page) - 1) * limit;
    const rows = this.stmts.listUsers.all(limit, offset);
    const total = this.stmts.countUsers.get().count;
    return {
      users: rows.map((r) => this._sanitizeUser(r)),
      page,
      limit,
      total,
      totalPages: Math.ceil(total / limit),
    };
  }

  getStats() {
    const tierCounts = this.stmts.countByTier.all().reduce((acc, row) => {
      acc[row.tier] = row.count;
      return acc;
    }, { free: 0, pro: 0, enterprise: 0 });

    return {
      total: this.stmts.countUsers.get().count,
      byTier: tierCounts,
    };
  }
}

module.exports = { UserService };
