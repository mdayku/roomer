import express from 'express';
import cors from 'cors';
import { detect } from './routes/detect';

const app = express();
app.use(cors());
app.use(express.json());
app.use('/api', detect);

const PORT = process.env.PORT || 4000;
app.listen(PORT, () => console.log(`API listening on :${PORT}`));

