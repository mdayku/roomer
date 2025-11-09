import express from 'express';
import cors from 'cors';
import { detect } from './routes/detect';

const app = express();

// CORS: API Gateway handles OPTIONS preflight, but Lambda must send headers for POST
// Enable CORS middleware for local dev, but in production Lambda sends headers directly
if (process.env.NODE_ENV !== 'production') {
  app.use(cors({
    origin: ['http://localhost:5173', 'http://localhost:3000'],
    credentials: true,
  }));
} else {
  // In production (Lambda), we set CORS headers in route handlers
  // This is because AWS_PROXY passes Lambda response headers through directly
}

// Handle JSON (for other endpoints if needed)
app.use(express.json({ limit: '10mb' }));
app.use(express.urlencoded({ extended: true, limit: '10mb' }));

app.use('/api', detect);

const PORT = process.env.PORT || 4000;
app.listen(PORT, () => {
  console.log(`========================================`);
  console.log(`API Server listening on port ${PORT}`);
  console.log(`========================================`);
  console.log(`Health check: http://localhost:${PORT}/api/health`);
  console.log(`Detect endpoint: http://localhost:${PORT}/api/detect`);
});

