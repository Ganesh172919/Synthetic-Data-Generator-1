import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

// Educational notes:
// - The React UI calls `/api/...` relative URLs.
// - In development, Vite proxies these requests to the Express server on port 3001.
// - This avoids CORS issues and keeps the client code environment-agnostic.
// - In production, you would typically serve the UI and API behind the same origin
//   (or configure a reverse proxy like nginx).

// https://vite.dev/config/
export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    proxy: {
      '/api': {
        target: 'http://localhost:3001',
        changeOrigin: true,
      }
    }
  }
})
