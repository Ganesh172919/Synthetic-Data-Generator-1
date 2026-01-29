# Synthetic Data Generator Server

Backend API server for the Universal Synthetic Dataset Generator Platform.

## Features

- **Python Integration**: Executes Python generators as subprocesses
- **Real-time Progress**: Streams generation progress via JSON events
- **Job Management**: Track, pause, and resume generation jobs
- **File Storage**: Manages generated datasets with download endpoints
- **Checkpoint Support**: Automatic checkpointing for fault tolerance
- **Template System**: Pre-configured templates for common domains

## Quick Start

### Prerequisites

- Node.js 16+ 
- Python 3.8+
- Required Python packages (see `../Pre-Work/universal_dataset_generator.py`)

### Installation

```bash
npm install
```

### Running the Server

```bash
# Development
npm run dev

# Production
npm start
```

The server will start on `http://localhost:3001`

## API Endpoints

### Generation

- `POST /api/generate` - Start a new generation job
- `GET /api/jobs/:jobId` - Get job status and progress
- `GET /api/jobs` - List all jobs
- `POST /api/jobs/:jobId/stop` - Stop a running job
- `GET /api/downloads/:jobId/:filename` - Download generated dataset

### Templates & Domains

- `GET /api/templates` - List available templates
- `GET /api/templates/:id` - Get template details
- `POST /api/domains` - Save custom domain configuration
- `GET /api/domains/:id` - Get domain configuration
- `GET /api/domains` - List all custom domains

### Health

- `GET /api/health` - Server health check

## Configuration

### Environment Variables

- `PORT` - Server port (default: 3001)
- `PYTHON_PATH` - Path to Python executable (default: python3)

### Generation Request

```json
{
  "domain": "technology",
  "targetCount": 1000,
  "batchSize": 25,
  "outputFormat": "jsonl",
  "domainDescription": "Programming and software development",
  "topics": ["Python", "JavaScript", "Algorithms"],
  "modelName": "mistralai/Mistral-7B-Instruct-v0.2",
  "temperature": 0.8,
  "useQuantization": true
}
```

## Architecture

```
Client Request
     ↓
Express API Server (server.js)
     ↓
Generator Runner (generator_runner.py)
     ↓
Universal Generator (Pre-Work/universal_dataset_generator.py)
     ↓
Mistral Model (Local/Colab)
     ↓
Generated Dataset (data/outputs/)
```

## Progress Events

The Python generator emits progress events in JSON format:

```json
{
  "type": "progress",
  "timestamp": "2026-01-29T00:00:00.000Z",
  "data": {
    "current": 500,
    "total": 1000,
    "rate": 10.5,
    "percentage": 50.0
  }
}
```

Event types:
- `init` - Generator initialization
- `start` - Generation started
- `progress` - Progress update
- `complete` - Generation completed
- `error` - Error occurred
- `shutdown` - Graceful shutdown

## File Structure

```
server/
├── server.js              # Main Express server
├── generator_runner.py    # Python subprocess wrapper
├── package.json          # Node.js dependencies
├── .gitignore           # Git ignore rules
└── data/                # Generated data (gitignored)
    ├── outputs/         # Dataset files
    ├── checkpoints/     # Checkpoint files
    └── configs/         # Job configurations
```

## Development

### Testing Python Integration

```bash
# Test the generator runner directly
python3 generator_runner.py --config '{"targetSize": 100, "batchSize": 10, "outputFile": "test_dataset", "outputFormat": "jsonl", "domainDescription": "Test domain", "topics": ["Test"]}' --generator universal
```

### Debugging

Enable verbose logging by checking server console output. All Python stdout/stderr is captured and logged.

## Production Deployment

### Docker Support (Future)

A Dockerfile will be provided for containerized deployment.

### Scaling Considerations

- Use a job queue (Bull, RabbitMQ) for multi-instance scaling
- Add Redis for shared job state
- Use S3/cloud storage for dataset files
- Implement WebSocket for real-time progress

## Security

- Input validation on all endpoints
- File path sanitization for downloads
- Process isolation for Python generators
- Rate limiting (recommended for production)

## License

MIT
