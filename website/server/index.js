const { start } = require('./src/server');

process.on('uncaughtException', (error) => {
  console.error('[FATAL] Uncaught exception:', error);
  process.exit(1);
});

process.on('unhandledRejection', (reason) => {
  console.error('[FATAL] Unhandled rejection:', reason);
  process.exit(1);
});

start().catch((error) => {
  console.error('[FATAL] Failed to start server:', error);
  process.exit(1);
});
