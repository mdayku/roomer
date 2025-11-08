import { Router } from 'express';
import multer from 'multer';
import { roomDetector, type DetectedRoom } from '../services/inference';

const upload = multer({ storage: multer.memoryStorage() });
export const detect = Router();

// POST /api/detect
// Body: multipart/form-data with 'image' field
// Returns: JSON array of DetectedRoom: [{id, bounding_box, name_hint}, ...]

// Note: CORS is handled by API Gateway, not here
// Removing explicit headers to avoid conflicts with API Gateway CORS configuration

detect.post('/detect', upload.fields([{ name: 'image' }, { name: 'model' }]), async (req, res) => {
  try {
    if (!req.files || !('image' in req.files) || req.files.image.length === 0) {
      // No image provided, return mock data in required format
      console.log('No image provided, returning mock detection');
      const mockRooms: DetectedRoom[] = [
        {
          id: 'room_001',
          bounding_box: [50, 50, 180, 240], // [x_min, y_min, x_max, y_max] in 0-1000 range
          name_hint: 'Entry Hall'
        },
        {
          id: 'room_002',
          bounding_box: [220, 60, 680, 450],
          name_hint: 'Main Office'
        }
      ];
      return res.json(mockRooms);
    }

    const imageFile = req.files.image[0];
    const selectedModel = Array.isArray(req.body.model) ? req.body.model[0] : req.body.model;

    console.log(`Processing image: ${imageFile.originalname} (${imageFile.size} bytes) with model: ${selectedModel || 'default'}`);

    // Use real inference with selected model
    const result = await roomDetector.detectRooms(imageFile.buffer, selectedModel);

    console.log(`Detection complete: ${result.length} rooms found`);
    res.json(result); // Returns JSON array: [{id, bounding_box, name_hint}, ...]

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

