import express from 'express';
import { detect } from './routes/detect';

const app = express();

// Note: CORS is handled by API Gateway when deployed
// Only enable CORS locally for development
if (process.env.NODE_ENV !== 'production') {
  const cors = require('cors');
  app.use(cors({
    origin: ['http://localhost:5173', 'http://localhost:3000'],
    credentials: true,
  }));
}

app.use(express.json());
app.use('/api', detect);

const PORT = process.env.PORT || 4000;
app.listen(PORT, () => console.log(`API listening on :${PORT}`));

