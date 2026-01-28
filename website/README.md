# 🧬 Synthetic Data Generator - Web Platform

A modern, full-stack web application for the Synthetic Data Generator platform.

## 🛠️ Tech Stack

- **Frontend**: React + Vite + TailwindCSS
- **Backend**: Node.js + Express
- **Icons**: Lucide React
- **Routing**: React Router

## 📁 Project Structure

```
website/
├── client/                 # React frontend
│   ├── src/
│   │   ├── components/     # Reusable UI components
│   │   ├── pages/          # Page components
│   │   ├── services/       # API service layer
│   │   ├── App.jsx         # Main app component
│   │   └── main.jsx        # Entry point
│   ├── package.json
│   └── vite.config.js
│
├── server/                 # Express backend
│   ├── index.js            # API server
│   └── package.json
│
└── README.md
```

## 🚀 Getting Started

### Prerequisites

- Node.js 18+ 
- npm or yarn

### Installation

1. **Install frontend dependencies:**
   ```bash
   cd website/client
   npm install
   ```

2. **Install backend dependencies:**
   ```bash
   cd website/server
   npm install
   ```

### Running the Application

**Option 1: Run both servers separately**

Terminal 1 - Backend:
```bash
cd website/server
npm start
```

Terminal 2 - Frontend:
```bash
cd website/client
npm run dev
```

**Option 2: Using concurrent processes**
```bash
# From website directory
npm run dev
```

The frontend will be available at `http://localhost:5173` and the API at `http://localhost:3001`.

## 🌐 Features

### Landing Page
- Hero section with key value propositions
- Feature highlights
- Performance benchmarks
- Testimonials
- Call-to-action sections

### Dashboard
- Real-time generation monitoring
- Progress tracking with live stats
- Job management
- Configuration panel
- Export functionality

### Templates
- Pre-built domain templates
- Search and filtering
- Template ratings and download counts
- Featured templates section

### Domain Builder
- Custom domain configuration
- Topic and subtopic management
- Question type selection
- Output settings customization
- Live preview panel

### Documentation
- Getting started guide
- Quick start instructions
- API reference
- Configuration options
- Output format specification

## 📡 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/health` | Health check |
| GET | `/api/templates` | List all templates |
| GET | `/api/templates/:id` | Get template by ID |
| POST | `/api/generate` | Start generation job |
| GET | `/api/jobs/:jobId` | Get job status |
| GET | `/api/jobs` | List all jobs |
| POST | `/api/domains` | Save custom domain |
| GET | `/api/domains/:id` | Get domain by ID |

## 🎨 UI Components

- **Navbar**: Responsive navigation with mobile menu
- **Footer**: Links and social icons
- **Cards**: Feature, template, and stat cards
- **Forms**: Configuration forms with validation
- **Progress**: Animated progress bars
- **Code Blocks**: Syntax highlighted with copy functionality

## 📦 Production Build

```bash
# Build frontend
cd website/client
npm run build

# The build output will be in dist/ directory
```

## 🔧 Configuration

### Environment Variables

Create a `.env` file in the server directory:

```env
PORT=3001
NODE_ENV=development
```

### Vite Configuration

The frontend is configured to proxy API requests to the backend during development. See `vite.config.js` for details.

## 📄 License

MIT License - see the [LICENSE](../LICENSE) file for details.
