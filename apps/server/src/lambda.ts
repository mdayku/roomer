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

// JSON body parser with increased limit for base64 images
app.use(express.json({ limit: '10mb' }));

// Minimal logging middleware
app.use((req, res, next) => {
  if (req.path === '/api/detect') {
    const imageLength = req.body?.image?.length || 0;
    console.log(`[LAMBDA] ${req.method} ${req.path} - image: ${imageLength} chars`);
  }
  next();
});

app.use('/api', detect);

// Wrap Express app for Lambda with error handling
const serverlessHandler = serverless(app);

export const handler = async (event: any, context: any) => {
  try {
    const result = await serverlessHandler(event, context);
    return result;
  } catch (error) {
    console.error('[LAMBDA HANDLER] Unhandled error:', error);
    const errorMessage = error instanceof Error ? error.message : 'Unknown error';
    return {
      statusCode: 500,
      headers: {
        'Access-Control-Allow-Origin': 'https://master.d7ra9ayxxa84o.amplifyapp.com',
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        error: 'Internal server error',
        message: errorMessage,
      }),
    };
  }
};

