'use strict';

const { NotFoundError, ConflictError, ValidationError } = require('../utils/errors');
const { generateId } = require('../utils/crypto');
const { safeJsonParse, nowIso } = require('../db');

/**
 * Plugin system infrastructure.
 * Manages plugin registration, lifecycle, and configuration.
 * Plugins can extend data generation with custom providers, parsers,
 * output formats, and domain-specific logic.
 */

const PLUGIN_HOOKS = [
  'beforeGeneration',
  'afterGeneration',
  'onBatchComplete',
  'onJobComplete',
  'transformOutput',
  'customProvider',
  'customParser',
  'customFormat',
];

class PluginService {
  constructor(db) {
    this.db = db;
    this.loadedPlugins = new Map();
    this.hooks = {};

    for (const hook of PLUGIN_HOOKS) {
      this.hooks[hook] = [];
    }

    this.stmts = {
      findById: db.prepare('SELECT * FROM plugins WHERE id = ?'),
      findByName: db.prepare('SELECT * FROM plugins WHERE name = ?'),
      listAll: db.prepare('SELECT * FROM plugins ORDER BY name'),
      listEnabled: db.prepare('SELECT * FROM plugins WHERE is_enabled = 1 ORDER BY name'),
      insert: db.prepare(`
        INSERT INTO plugins (id, name, version, description, author, entry_point, is_enabled, config_json, installed_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
      `),
      update: db.prepare(`
        UPDATE plugins SET version = ?, description = ?, config_json = ?, updated_at = ?
        WHERE id = ?
      `),
      toggle: db.prepare('UPDATE plugins SET is_enabled = ?, updated_at = ? WHERE id = ?'),
      remove: db.prepare('DELETE FROM plugins WHERE id = ?'),
    };
  }

  _toApi(row) {
    if (!row) return null;
    return {
      id: row.id,
      name: row.name,
      version: row.version,
      description: row.description,
      author: row.author,
      entryPoint: row.entry_point,
      isEnabled: Boolean(row.is_enabled),
      config: safeJsonParse(row.config_json, {}),
      installedAt: row.installed_at,
      updatedAt: row.updated_at,
    };
  }

  /**
   * Register a new plugin.
   */
  register({ name, version, description, author, entryPoint, config = {} }) {
    if (!name || typeof name !== 'string' || name.trim().length < 1) {
      throw new ValidationError('Plugin name is required');
    }
    if (!version || typeof version !== 'string') {
      throw new ValidationError('Plugin version is required');
    }
    if (!entryPoint || typeof entryPoint !== 'string') {
      throw new ValidationError('Plugin entry point is required');
    }

    const existing = this.stmts.findByName.get(name.trim());
    if (existing) {
      throw new ConflictError(`Plugin '${name}' is already registered`);
    }

    const id = generateId('plg');
    const now = nowIso();
    const configJson = JSON.stringify(config);

    this.stmts.insert.run(
      id, name.trim(), version.trim(), description || null,
      author || null, entryPoint.trim(), 0, configJson, now, now
    );

    return this._toApi(this.stmts.findById.get(id));
  }

  /**
   * Enable a plugin.
   */
  enable(pluginId) {
    const row = this.stmts.findById.get(pluginId);
    if (!row) throw new NotFoundError('Plugin');

    this.stmts.toggle.run(1, nowIso(), pluginId);
    return this._toApi(this.stmts.findById.get(pluginId));
  }

  /**
   * Disable a plugin.
   */
  disable(pluginId) {
    const row = this.stmts.findById.get(pluginId);
    if (!row) throw new NotFoundError('Plugin');

    this.stmts.toggle.run(0, nowIso(), pluginId);
    this.loadedPlugins.delete(row.name);
    return this._toApi(this.stmts.findById.get(pluginId));
  }

  /**
   * Uninstall a plugin.
   */
  uninstall(pluginId) {
    const row = this.stmts.findById.get(pluginId);
    if (!row) throw new NotFoundError('Plugin');

    this.loadedPlugins.delete(row.name);
    this.stmts.remove.run(pluginId);
    return true;
  }

  listAll() {
    return this.stmts.listAll.all().map((r) => this._toApi(r));
  }

  listEnabled() {
    return this.stmts.listEnabled.all().map((r) => this._toApi(r));
  }

  getById(pluginId) {
    const row = this.stmts.findById.get(pluginId);
    if (!row) throw new NotFoundError('Plugin');
    return this._toApi(row);
  }

  /**
   * Update a plugin configuration.
   */
  updateConfig(pluginId, { version, description, config }) {
    const row = this.stmts.findById.get(pluginId);
    if (!row) throw new NotFoundError('Plugin');

    const newVersion = version || row.version;
    const newDescription = description !== undefined ? description : row.description;
    const newConfig = config !== undefined ? JSON.stringify(config) : row.config_json;

    this.stmts.update.run(newVersion, newDescription, newConfig, nowIso(), pluginId);
    return this._toApi(this.stmts.findById.get(pluginId));
  }

  /**
   * Execute a plugin hook, calling all registered handlers.
   */
  async executeHook(hookName, context = {}) {
    if (!this.hooks[hookName]) return context;

    for (const handler of this.hooks[hookName]) {
      try {
        const result = await handler(context);
        if (result !== undefined) {
          Object.assign(context, result);
        }
      } catch (err) {
        context.pluginErrors = context.pluginErrors || [];
        context.pluginErrors.push({
          hook: hookName,
          plugin: handler.pluginName || 'unknown',
          error: err.message,
        });
      }
    }
    return context;
  }

  /**
   * Get available hooks that plugins can use.
   */
  getAvailableHooks() {
    return PLUGIN_HOOKS;
  }
}

module.exports = { PluginService, PLUGIN_HOOKS };
