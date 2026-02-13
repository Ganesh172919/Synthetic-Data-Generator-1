import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App.jsx'

// Educational note:
// React StrictMode is enabled in development to help surface unsafe side effects.
// Some lifecycle-like effects may run twice in dev (not in production), so avoid
// writing code that assumes effects run exactly once.

createRoot(document.getElementById('root')).render(
  <StrictMode>
    <App />
  </StrictMode>,
)
