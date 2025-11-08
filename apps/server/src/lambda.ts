/**
 * AWS Lambda handler wrapper for Express server
 * This file adapts the Express app to work as a Lambda function
 */
import serverless from 'serverless-http';
import express from 'express';
import { detect } from './routes/detect';

const app = express();

// CORS: API Gateway handles OPTIONS preflight, but Lambda must send headers for POST
// Enable CORS middleware for local dev, but in production Lambda sends headers directly
if (process.env.NODE_ENV !== 'production') {
  const cors = require('cors');
  app.use(cors({
    origin: ['http://localhost:5173', 'http://localhost:3000'],
    credentials: true,
  }));
} else {
  // In production (Lambda), we set CORS headers in route handlers
  // This is because AWS_PROXY passes Lambda response headers through directly
}

app.use(express.json({ limit: '10mb' })); // Increase limit for base64 images
app.use('/api', detect);

// Wrap Express app for Lambda
export const handler = serverless(app);

