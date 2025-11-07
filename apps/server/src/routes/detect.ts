import { Router } from 'express';
import multer from 'multer';
import { roomDetector } from '../services/inference';
import { mockDetect } from '../services/detector';

const upload = multer({ storage: multer.memoryStorage() });
export const detect = Router();

// POST /api/detect
// Body: multipart/form-data with 'image' field
// Returns: GeoJSON FeatureCollection

detect.post('/detect', upload.single('image'), async (req, res) => {
  try {
    if (!req.file) {
      // No image provided, return mock data
      console.log('No image provided, returning mock detection');
      const fc = mockDetect();
      return res.json(fc);
    }

    console.log(`Processing image: ${req.file.originalname} (${req.file.size} bytes)`);

    // Use real inference
    const result = await roomDetector.detectRooms(req.file.buffer);

    console.log(`Detection complete: ${result.features.length} rooms found`);
    res.json(result);

  } catch (error) {
    console.error('Detection error:', error);
    res.status(500).json({
      error: 'Detection failed',
      message: error instanceof Error ? error.message : 'Unknown error'
    });
  }
});

// GET /api/health

detect.get('/health', (_, res) => res.json({ ok: true }));

