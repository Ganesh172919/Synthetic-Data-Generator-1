/**
 * ESLint (Flat Config) for the React client.
 *
 * Educational notes:
 * - This repo uses ESLint's "flat config" (`eslint.config.js`) rather than legacy `.eslintrc`.
 * - We enable React Hooks rules and React Refresh rules for a better dev experience.
 * - `globalIgnores(['dist'])` prevents linting build artifacts.
 *
 * How to run:
 *   cd website/client
 *   npm run lint
 */

import js from '@eslint/js'
import globals from 'globals'
import reactHooks from 'eslint-plugin-react-hooks'
import reactRefresh from 'eslint-plugin-react-refresh'
import { defineConfig, globalIgnores } from 'eslint/config'

export default defineConfig([
  globalIgnores(['dist']),
  {
    files: ['**/*.{js,jsx}'],
    extends: [
      js.configs.recommended,
      reactHooks.configs.flat.recommended,
      reactRefresh.configs.vite,
    ],
    languageOptions: {
      ecmaVersion: 2020,
      globals: globals.browser,
      parserOptions: {
        ecmaVersion: 'latest',
        ecmaFeatures: { jsx: true },
        sourceType: 'module',
      },
    },
    rules: {
      'no-unused-vars': ['error', { varsIgnorePattern: '^[A-Z_]' }],
    },
  },
])
