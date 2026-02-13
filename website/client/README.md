# React + Vite

This template provides a minimal setup to get React working in Vite with HMR and some ESLint rules.

Currently, two official plugins are available:

- [@vitejs/plugin-react](https://github.com/vitejs/vite-plugin-react/blob/main/packages/plugin-react) uses [Babel](https://babeljs.io/) (or [oxc](https://oxc.rs) when used in [rolldown-vite](https://vite.dev/guide/rolldown)) for Fast Refresh
- [@vitejs/plugin-react-swc](https://github.com/vitejs/vite-plugin-react/blob/main/packages/plugin-react-swc) uses [SWC](https://swc.rs/) for Fast Refresh

## React Compiler

The React Compiler is not enabled on this template because of its impact on dev & build performances. To add it, see [this documentation](https://react.dev/learn/react-compiler/installation).

## Expanding the ESLint configuration

If you are developing a production application, we recommend using TypeScript with type-aware lint rules enabled. Check out the [TS template](https://github.com/vitejs/vite/tree/main/packages/create-vite/template-react-ts) for information on how to integrate TypeScript and [`typescript-eslint`](https://typescript-eslint.io) in your project.

## Educational Notes (Added)

### What this folder is for

`website/client/` is the React + Vite frontend for the Synthetic Data Generator web platform.

It focuses on:
- UX (dashboard, templates, domain builder)
- a small reusable component library (`src/components/ui`)
- client-side routing (React Router)
- theme management + polished visuals (design tokens + light/dark mode)

### Reality-aligned updates (how it talks to the backend)

In development, the client makes requests to `/api/...`. Vite proxies these calls to the Express server:

- Client dev server: `http://localhost:5173`
- API server: `http://localhost:3001`
- Proxy config: `vite.config.js`

This avoids common CORS problems during local dev.

### How pages map to features

Routes are defined in `src/App.jsx`:

- `/` → Landing page (marketing + CTA)
- `/dashboard` → Start jobs + monitor progress + export
- `/templates` → Browse templates
- `/domain-builder` → Create and save custom domain configs
- `/documentation` → In-app documentation view

### Demo-mode fallbacks (important for learning)

This UI has a few “fallbacks” so it still looks good even if the backend is down:

- Templates page can fall back to a local `defaultTemplates` list.
- Dashboard can simulate progress in the UI if the API call fails.

These are great for demos, but keep in mind they can drift from real backend state if you later implement a true job runner.

### Edge cases & failure modes

- React `StrictMode` (enabled in `src/main.jsx`) can double-invoke some effects in development.
- Network failures should surface as user-visible toasts; ensure API errors are handled consistently.
- Large datasets should be streamed/downloaded from the backend in production; in the demo server downloads are in-memory payloads.

### Next steps / exercises

1. Replace direct `fetch('/api/...')` calls with the shared wrapper in `src/services/api.js` for consistency.
2. Convert the dashboard to poll real job status from the server (no simulated timer) and document the resulting state machine.
3. Add an accessibility checklist for modals, toasts, and form inputs.
