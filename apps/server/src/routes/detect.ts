import { Router } from 'express';
import multer from 'multer';
import { roomDetector } from '../services/inference';
import { mockDetect } from '../services/detector';

const upload = multer({ storage: multer.memoryStorage() });
export const detect = Router();

// POST /api/detect
// Body: multipart/form-data with 'image' field
// Returns: GeoJSON FeatureCollection

detect.post('/detect', upload.fields([{ name: 'image' }, { name: 'model' }]), async (req, res) => {
  try {
    if (!req.files || !('image' in req.files) || req.files.image.length === 0) {
      // No image provided, return mock data
      console.log('No image provided, returning mock detection');
      const fc = mockDetect();
      return res.json(fc);
    }

    const imageFile = req.files.image[0];
    const selectedModel = Array.isArray(req.body.model) ? req.body.model[0] : req.body.model;

    console.log(`Processing image: ${imageFile.originalname} (${imageFile.size} bytes) with model: ${selectedModel || 'default'}`);

    // Use real inference with selected model
    const result = await roomDetector.detectRooms(imageFile.buffer, selectedModel);

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

