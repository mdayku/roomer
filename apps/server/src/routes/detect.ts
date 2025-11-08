import { Router } from 'express';
import multer from 'multer';
import { roomDetector, type DetectedRoom } from '../services/inference';

const upload = multer({ storage: multer.memoryStorage() });
export const detect = Router();

// Simple multipart parser for API Gateway binary data
function parseMultipart(buffer: Buffer, boundary: string): Array<{ name: string; data: Buffer }> {
  const parts: Array<{ name: string; data: Buffer }> = [];
  const boundaryBuffer = Buffer.from(`--${boundary}`);
  const endBoundaryBuffer = Buffer.from(`--${boundary}--`);
  
  let start = 0;
  while (true) {
    const boundaryIndex = buffer.indexOf(boundaryBuffer, start);
    if (boundaryIndex === -1) break;
    
    const nextBoundaryIndex = buffer.indexOf(boundaryBuffer, boundaryIndex + boundaryBuffer.length);
    if (nextBoundaryIndex === -1) break;
    
    const partBuffer = buffer.slice(boundaryIndex + boundaryBuffer.length, nextBoundaryIndex);
    const headerEnd = partBuffer.indexOf(Buffer.from('\r\n\r\n'));
    if (headerEnd === -1) {
      start = nextBoundaryIndex;
      continue;
    }
    
    const headers = partBuffer.slice(0, headerEnd).toString('utf-8');
    const contentDisposition = headers.match(/Content-Disposition:.*name="([^"]+)"/i);
    if (contentDisposition) {
      const name = contentDisposition[1];
      const data = partBuffer.slice(headerEnd + 4);
      // Remove trailing \r\n
      const cleanData = data.slice(0, data.length - 2);
      parts.push({ name, data: cleanData });
    }
    
    start = nextBoundaryIndex;
  }
  
  return parts;
}

// POST /api/detect
// Body: multipart/form-data with 'image' field
// Returns: JSON array of DetectedRoom: [{id, bounding_box, name_hint}, ...]

// With AWS_PROXY integration, Lambda response headers pass through directly
// We need to send CORS headers from Lambda for POST responses
// OPTIONS preflight is handled by API Gateway MOCK integration

// Handle multipart/form-data manually for API Gateway
detect.post('/detect', async (req, res) => {
  console.log('[DETECT] Request received');
  console.log('[DETECT] Headers:', JSON.stringify(req.headers, null, 2));
  console.log('[DETECT] Content-Type:', req.headers['content-type']);
  console.log('[DETECT] Body type:', typeof req.body);
  console.log('[DETECT] Body is buffer:', Buffer.isBuffer(req.body));
  
  // Check if body is a raw buffer or string (from API Gateway binary handling)
  let imageBuffer: Buffer | undefined;
  let modelId: string | undefined;
  
  let bodyBuffer: Buffer;
  if (Buffer.isBuffer(req.body)) {
    bodyBuffer = req.body;
  } else if (typeof req.body === 'string') {
    // Body might be base64 encoded string from API Gateway
    try {
      bodyBuffer = Buffer.from(req.body, 'base64');
      console.log('[DETECT] Decoded base64 body:', bodyBuffer.length, 'bytes');
    } catch {
      bodyBuffer = Buffer.from(req.body, 'utf-8');
    }
  } else {
    // Try multer parsing as fallback for regular requests
    console.log('[DETECT] Body is not buffer/string, using multer');
    return upload.fields([{ name: 'image' }, { name: 'model' }])(req, res, async () => {
      await handleDetectRequest(req, res);
    });
  }
  
  // Parse multipart manually
  const contentType = req.headers['content-type'] || '';
  console.log('[DETECT] Content-Type:', contentType);
  const boundary = contentType.match(/boundary=([^;\s]+)/)?.[1];
  
  if (boundary) {
    console.log('[DETECT] Found boundary:', boundary);
    // Parse multipart manually
    const parts = parseMultipart(bodyBuffer, boundary);
    console.log('[DETECT] Parsed', parts.length, 'parts:', parts.map(p => p.name));
    const imagePart = parts.find(p => p.name === 'image');
    const modelPart = parts.find(p => p.name === 'model');
    
    if (imagePart) {
      imageBuffer = imagePart.data;
      console.log('[DETECT] Extracted image from multipart:', imageBuffer.length, 'bytes');
    }
    if (modelPart) {
      modelId = modelPart.data.toString('utf-8');
      console.log('[DETECT] Extracted model:', modelId);
    }
  } else {
    console.log('[DETECT] No boundary found in Content-Type');
    // If no boundary, maybe the body is just the image
    imageBuffer = bodyBuffer;
  }
  
  // Continue with the request handling
  await handleDetectRequest(req, res, imageBuffer, modelId);
});

async function handleDetectRequest(req: any, res: any, imageBuffer?: Buffer, modelId?: string) {
  // Set CORS headers for POST response (required with AWS_PROXY)
  const origin = req.headers.origin;
  const allowedOrigin = 'https://master.d7ra9ayxxa84o.amplifyapp.com';
  
  // Always set CORS headers - use specific origin if it matches, otherwise use the request origin
  if (origin === allowedOrigin) {
    res.header('Access-Control-Allow-Origin', origin);
  } else if (origin) {
    // For debugging: log unexpected origins
    console.log(`[CORS] Unexpected origin: ${origin}, expected: ${allowedOrigin}`);
    res.header('Access-Control-Allow-Origin', origin);  // Allow the requesting origin
  } else {
    // No origin header (same-origin request or server-to-server)
    res.header('Access-Control-Allow-Origin', allowedOrigin);
  }
  
  res.header('Access-Control-Allow-Methods', 'POST, OPTIONS, GET');
  res.header('Access-Control-Allow-Headers', 'Content-Type, Authorization, X-Amz-Date, X-Api-Key, X-Amz-Security-Token');
  res.header('Access-Control-Allow-Credentials', 'true');
  
  try {
    // Check if we have image from manual parsing or from multer
    let finalImageBuffer: Buffer | undefined = imageBuffer;
    let finalModelId: string | undefined = modelId;
    
    if (!finalImageBuffer && req.files && 'image' in req.files && req.files.image.length > 0) {
      // Fallback to multer-parsed files
      finalImageBuffer = req.files.image[0].buffer;
      finalModelId = Array.isArray(req.body.model) ? req.body.model[0] : req.body.model;
    }
    
    if (!finalImageBuffer || finalImageBuffer.length === 0) {
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

    console.log(`Processing image: ${finalImageBuffer.length} bytes with model: ${finalModelId || 'default'}`);

    // Use real inference with selected model
    const result = await roomDetector.detectRooms(finalImageBuffer, finalModelId);

    console.log(`Detection complete: ${result.length} rooms found`);
    res.json(result); // Returns JSON array: [{id, bounding_box, name_hint}, ...]

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
}

// GET /api/health

detect.get('/health', (_, res) => res.json({ ok: true }));

