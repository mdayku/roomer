import { Router } from 'express';
import multer from 'multer';
import { mockDetect } from '../services/detector';

const upload = multer({ storage: multer.memoryStorage() });
export const detect = Router();

// POST /api/detect (image optional for mock)
// returns GeoJSON FeatureCollection

detect.post('/detect', upload.single('image'), async (req, res) => {
  const fc = mockDetect();
  res.json(fc);
});

// GET /api/health

detect.get('/health', (_, res) => res.json({ ok: true }));

