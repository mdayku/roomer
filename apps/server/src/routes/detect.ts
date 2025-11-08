import { Router } from 'express';
import { roomDetector, type DetectedRoom } from '../services/inference';

export const detect = Router();

// POST /api/detect
// Body: JSON with { image: base64String, model?: string }
// Returns: JSON array of DetectedRoom: [{id, bounding_box, name_hint}, ...]

detect.post('/detect', async (req, res) => {
  console.log('[DETECT] Request received');
  console.log('[DETECT] Body type:', typeof req.body);
  console.log('[DETECT] Has image:', !!req.body?.image);
  
  // Set CORS headers for POST response (required with AWS_PROXY)
  const origin = req.headers.origin;
  const allowedOrigin = 'https://master.d7ra9ayxxa84o.amplifyapp.com';
  
  if (origin === allowedOrigin) {
    res.header('Access-Control-Allow-Origin', origin);
  } else if (origin) {
    console.log(`[CORS] Unexpected origin: ${origin}, expected: ${allowedOrigin}`);
    res.header('Access-Control-Allow-Origin', origin);
  } else {
    res.header('Access-Control-Allow-Origin', allowedOrigin);
  }
  
  res.header('Access-Control-Allow-Methods', 'POST, OPTIONS, GET');
  res.header('Access-Control-Allow-Headers', 'Content-Type, Authorization, X-Amz-Date, X-Api-Key, X-Amz-Security-Token');
  res.header('Access-Control-Allow-Credentials', 'true');
  
  try {
    const { image: imageBase64, model: modelId } = req.body || {};
    
    if (!imageBase64) {
      // No image provided, return mock data
      console.log('No image provided, returning mock detection');
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
      return res.json(mockRooms);
    }

    // Convert base64 to buffer
    const imageBuffer = Buffer.from(imageBase64, 'base64');
    console.log(`Processing image: ${imageBuffer.length} bytes with model: ${modelId || 'default'}`);

    // Use real inference with selected model
    const result = await roomDetector.detectRooms(imageBuffer, modelId);

    console.log(`Detection complete: ${result.length} rooms found`);
    res.json(result);

  } catch (error) {
    console.error('Detection error:', error);
    const errorMessage = error instanceof Error ? error.message : 'Unknown error';
    const errorStack = error instanceof Error ? error.stack : undefined;
    console.error('Error stack:', errorStack);
    res.status(500).json({
      error: 'Detection failed',
      message: errorMessage,
      stack: process.env.NODE_ENV === 'production' ? undefined : errorStack
    });
  }
});

// GET /api/health

detect.get('/health', (_, res) => res.json({ ok: true }));

