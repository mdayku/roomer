# Electron Integration Guide

## Overview
This guide shows how to integrate the room detection feature into your Electron video editor app.

## Architecture

```
Electron App
├── Main Process (Node.js)
│   ├── Room Detection Service (inference.ts)
│   └── IPC Handlers
├── Renderer Process (UI)
│   └── Calls IPC to detect rooms
└── Python Runtime
    ├── inference.py
    └── Models (best.pt files)
```

## Step 1: Copy Required Files

Copy these files/folders to your Electron app:

```
your-electron-app/
├── src/
│   └── services/
│       ├── roomDetection.ts      # Copy from apps/server/src/services/inference.ts
│       └── inference.py           # Copy from apps/server/src/services/inference.py
├── models/                       # Copy trained models
│   └── yolo-v8l-200epoch/
│       └── weights/
│           └── best.pt
└── package.json
```

## Step 2: Install Dependencies

Add to your Electron app's `package.json`:

```json
{
  "dependencies": {
    // ... your existing deps
  },
  "devDependencies": {
    // ... your existing deps
  }
}
```

**Note:** No additional npm dependencies needed! The inference service only uses Node.js built-ins (`child_process`, `path`, `fs`).

## Step 3: Set Up Python Environment

### Option A: Bundle Python (Recommended for Distribution)

Use `pyinstaller` or similar to bundle Python + dependencies:

```bash
# In your Electron app directory
pip install pyinstaller
pyinstaller --onefile --name room-detection inference.py
```

Then update `roomDetection.ts` to use the bundled executable.

### Option B: Require Python Installation

Ensure Python 3.8+ with these packages is installed:
- `ultralytics`
- `Pillow`
- `numpy`

## Step 4: Create IPC Handler in Main Process

In your Electron main process file (e.g., `main.js` or `main.ts`):

```typescript
import { ipcMain } from 'electron';
import { RoomDetectionInference } from './services/roomDetection';
import { readFileSync } from 'fs';

// Initialize the inference service
const roomDetector = new RoomDetectionInference();

// IPC handler for room detection
ipcMain.handle('detect-rooms', async (event, imagePath: string, modelId?: string) => {
  try {
    // Read image file
    const imageBuffer = readFileSync(imagePath);
    
    // Run detection
    const result = await roomDetector.detectRooms(imageBuffer, modelId);
    
    return {
      success: true,
      data: result
    };
  } catch (error) {
    console.error('Room detection failed:', error);
    return {
      success: false,
      error: error instanceof Error ? error.message : 'Unknown error'
    };
  }
});
```

## Step 5: Update Renderer Process

In your renderer process (UI code):

```typescript
import { ipcRenderer } from 'electron';

export async function detectRooms(imagePath: string, modelId?: string) {
  const result = await ipcRenderer.invoke('detect-rooms', imagePath, modelId);
  
  if (!result.success) {
    throw new Error(result.error);
  }
  
  return result.data; // { detections: [...], annotated_image: "base64..." }
}
```

## Step 6: Update Path Resolution

Update `roomDetection.ts` to work in Electron's packaged environment:

```typescript
import { app } from 'electron';
import { join } from 'path';

class RoomDetectionInference {
  constructor() {
    // Use app.getAppPath() in Electron
    const appPath = app.isPackaged 
      ? process.resourcesPath 
      : app.getAppPath();
    
    // Find Python executable
    const venvPython = process.platform === 'win32'
      ? join(appPath, '.venv', 'Scripts', 'python.exe')
      : join(appPath, '.venv', 'bin', 'python3');
    
    this.pythonPath = existsSync(venvPython) 
      ? venvPython 
      : (process.platform === 'win32' ? 'python' : 'python3');
    
    // Find inference script
    this.inferenceScriptPath = join(appPath, 'src', 'services', 'inference.py');
    
    // Update Python script to find models relative to app path
    // Models should be at: appPath/models/yolo-v8l-200epoch/weights/best.pt
  }
}
```

## Step 7: Update Python Script for Electron

Update `inference.py` to find models in Electron's resource path:

```python
import sys
import os
from pathlib import Path

# In Electron, models are bundled with the app
if getattr(sys, 'frozen', False):
    # Running as bundled executable
    APP_PATH = Path(sys._MEIPASS)
else:
    # Running as script
    APP_PATH = Path.cwd()

MODELS_DIR = APP_PATH / 'models'
PROJECT_ROOT = APP_PATH  # Or adjust based on your structure

MODELS = {
    'yolo-v8l-200epoch': {
        'local_path': MODELS_DIR / 'yolo-v8l-200epoch' / 'weights' / 'best.pt',
        'description': 'YOLO v8 Large - 200 epochs (Best Model)'
    },
    # ... other models
}
```

## Step 8: Bundle Models in Electron Build

Update your Electron build config (e.g., `electron-builder`):

```json
{
  "build": {
    "extraResources": [
      {
        "from": "models",
        "to": "models",
        "filter": ["**/*"]
      },
      {
        "from": "src/services/inference.py",
        "to": "inference.py"
      }
    ]
  }
}
```

## Step 9: Usage Example

```typescript
// In your video editor UI
async function detectRoomsInBlueprint(imagePath: string) {
  try {
    const result = await detectRooms(imagePath, 'yolo-v8l-200epoch');
    
    // result.detections: Array of { id, bounding_box, name_hint }
    // result.annotated_image: Base64 PNG with bounding boxes drawn
    
    // Display annotated image
    const img = document.createElement('img');
    img.src = `data:image/png;base64,${result.annotated_image}`;
    document.body.appendChild(img);
    
    // Use detections
    console.log(`Found ${result.detections.length} rooms`);
    result.detections.forEach(room => {
      console.log(`Room ${room.id}: ${room.bounding_box}`);
    });
  } catch (error) {
    console.error('Detection failed:', error);
  }
}
```

## Alternative: Simpler Integration (No Python)

If you want to avoid Python dependencies, you could:

1. **Use ONNX Runtime** - Convert YOLO model to ONNX and use `onnxruntime-node`
2. **Use TensorFlow.js** - Convert to TensorFlow.js format
3. **Call External API** - Keep the web server running and call it via HTTP

## Troubleshooting

### Python Not Found
- Ensure Python is in PATH or bundle it with your app
- Check `pythonPath` resolution in `roomDetection.ts`

### Model Not Found
- Verify model paths in `inference.py` match your Electron bundle structure
- Use `app.getAppPath()` or `process.resourcesPath` for correct paths

### Permission Issues
- On macOS, you may need to sign the Python executable
- Consider bundling Python with your app to avoid system dependencies

## File Size Considerations

- Python runtime: ~50-100MB
- ultralytics + dependencies: ~200-300MB
- Model files: ~50-200MB each
- **Total: ~300-600MB** (can be reduced with model quantization)

Consider:
- Using smaller models (yolo-v8s instead of yolo-v8l)
- Model quantization/pruning
- Lazy loading of Python runtime


