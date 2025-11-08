import express from 'express';
import cors from 'cors';
import { detect } from './routes/detect';

const app = express();

// Configure CORS with explicit origin for Amplify deployment
const corsOptions = {
  origin: [
    'https://master.d7ra9ayxxa84o.amplifyapp.com',
    'http://localhost:5173',
    'http://localhost:3000',
  ],
  credentials: true,
  methods: ['GET', 'POST', 'OPTIONS'],
  allowedHeaders: ['Content-Type', 'Authorization', 'X-Amz-Date', 'X-Api-Key', 'X-Amz-Security-Token'],
};

app.use(cors(corsOptions));

// Handle preflight OPTIONS requests explicitly
app.options('*', cors(corsOptions));

app.use(express.json());
app.use('/api', detect);

const PORT = process.env.PORT || 4000;
app.listen(PORT, () => console.log(`API listening on :${PORT}`));

