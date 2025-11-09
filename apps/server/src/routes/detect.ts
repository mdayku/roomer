import { Router } from 'express';
import multer from 'multer';
import { roomDetector, type DetectedRoom } from '../services/inference';

export const detect = Router();

// Multer for file uploads
const upload = multer({ 
  storage: multer.memoryStorage(),
  limits: { fileSize: 10 * 1024 * 1024 } // 10MB
});

// POST /api/detect
// Body: multipart/form-data with image file and optional model string
// Returns: JSON array of DetectedRoom: [{id, bounding_box, name_hint}, ...]

detect.post('/detect', upload.single('image'), async (req, res) => {
  console.log('[DETECT] Request received');
  console.log('[DETECT] Has file:', !!req.file);
  console.log('[DETECT] File size:', req.file?.size);
  console.log('[DETECT] Model:', req.body?.model);
  
  // Set CORS headers
  const origin = req.headers.origin;
  if (origin) {
    res.header('Access-Control-Allow-Origin', origin);
  } else {
    res.header('Access-Control-Allow-Origin', '*');
  }
  
  res.header('Access-Control-Allow-Methods', 'POST, OPTIONS, GET');
  res.header('Access-Control-Allow-Headers', 'Content-Type, Authorization');
  res.header('Access-Control-Allow-Credentials', 'true');
  
  try {
    const file = req.file;
    const modelId = req.body?.model;
    
    if (!file) {
      console.log('[DETECT] No image provided, returning mock data');
      const mockRooms: DetectedRoom[] = [
        {
          id: 'room_001',
          bounding_box: [50, 50, 180, 240],
          name_hint: 'Entry Hall'
        },
        {
          id: 'room_002',
          bounding_box: [220, 60, 680, 450],
          name_hint: 'Main Office'
        }
      ];
      return res.json({ detections: mockRooms, annotated_image: null });
    }

    console.log(`[DETECT] Processing ${file.size} byte image (${file.mimetype}) with model: ${modelId || 'default'}`);

    // Use real inference with selected model
    console.log('[DETECT] Calling roomDetector.detectRooms...');
    const result = await roomDetector.detectRooms(file.buffer, modelId);
    console.log(`[DETECT] Found ${result.detections?.length || 0} rooms`);
    
    // Return both detections and annotated image
    res.json({
      detections: result.detections || [],
      annotated_image: result.annotated_image || null
    });

  } catch (error) {
    console.error('[DETECT] Error:', error);
    const errorMessage = error instanceof Error ? error.message : 'Unknown error';
    console.error('[DETECT] Error stack:', error instanceof Error ? error.stack : 'No stack');
    res.status(500).json({
      error: 'Detection failed',
      message: errorMessage
    });
  }
});

// GET /api/health

detect.get('/health', (_, res) => res.json({ ok: true }));

