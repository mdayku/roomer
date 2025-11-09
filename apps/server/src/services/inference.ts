import { spawn } from 'child_process';
import { join, dirname } from 'path';
import { existsSync } from 'fs';
import { fileURLToPath } from 'url';

// DetectedRoom type matching the required output schema
export interface DetectedRoom {
  id: string;
  bounding_box: [number, number, number, number]; // [x_min, y_min, x_max, y_max] in 0-1000 range
  name_hint: string;
}

// Get __dirname equivalent for ES modules
const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

class RoomDetectionInference {
  private pythonPath: string;
  private inferenceScriptPath: string;

  constructor() {
    // Use python3 on Unix, python on Windows
    // Try venv python first if it exists
    const venvPython = process.platform === 'win32' 
      ? join(process.cwd(), '.venv', 'Scripts', 'python.exe')
      : join(process.cwd(), '.venv', 'bin', 'python3');
    
    this.pythonPath = existsSync(venvPython) ? venvPython : (process.platform === 'win32' ? 'python' : 'python3');
    
    // Path to Python inference script - try multiple possible locations
    const possiblePaths = [
      join(__dirname, 'inference.py'),
      join(process.cwd(), 'apps', 'server', 'src', 'services', 'inference.py'),
      join(__dirname, '..', 'services', 'inference.py'),
      '/var/task/services/inference.py'
    ];
    
    this.inferenceScriptPath = possiblePaths.find(p => existsSync(p)) || possiblePaths[0];
    console.log(`[INFERENCE] Using Python: ${this.pythonPath}`);
    console.log(`[INFERENCE] Using Python script: ${this.inferenceScriptPath}`);
  }

  async detectRooms(imageBuffer: Buffer, modelId?: string): Promise<{ detections: DetectedRoom[], annotated_image?: string }> {
    // Use local Python inference instead of SageMaker
    console.log(`[INFERENCE] Running local inference (model: ${modelId || 'default'})`);
    
    try {
      return await this.runLocalInference(imageBuffer, modelId);
    } catch (error) {
      console.error('[INFERENCE] Local inference failed:', error);
      console.log('[INFERENCE] Falling back to mock detection');
      return { detections: this.getMockRooms() };
    }
  }

  private async runLocalInference(imageBuffer: Buffer, modelId?: string): Promise<{ detections: DetectedRoom[], annotated_image?: string }> {
    return new Promise((resolve, reject) => {
      const modelIdStr = modelId || 'default';
      const modelIdBytes = Buffer.from(modelIdStr, 'utf-8');
      
      // Send: [4 bytes: model ID length][model ID bytes][image buffer]
      const modelIdLength = Buffer.allocUnsafe(4);
      modelIdLength.writeUInt32BE(modelIdBytes.length, 0);

      // Set working directory to project root so Python can find models
      const projectRoot = process.cwd().includes('apps') 
        ? join(process.cwd(), '..', '..') 
        : process.cwd();
      
      const pythonProcess = spawn(this.pythonPath, [this.inferenceScriptPath], {
        cwd: projectRoot,  // Run from project root so Path.cwd() works like comprehensive_test.py
        stdio: ['pipe', 'pipe', 'pipe']
      });

      let stdout = '';
      let stderr = '';

      pythonProcess.stdout.on('data', (data) => {
        stdout += data.toString();
      });

      pythonProcess.stderr.on('data', (data) => {
        stderr += data.toString();
        // Log Python stderr in real-time for debugging
        console.error(`[PYTHON STDERR] ${data.toString()}`);
      });

      pythonProcess.on('close', (code) => {
        console.log(`[INFERENCE] Python process exited with code ${code}`);
        console.log(`[INFERENCE] stdout length: ${stdout.length} chars`);
        console.log(`[INFERENCE] stderr length: ${stderr.length} chars`);
        
        if (code !== 0) {
          console.error(`[INFERENCE] Python script exited with code ${code}`);
          console.error(`[INFERENCE] stderr: ${stderr}`);
          console.error(`[INFERENCE] stdout (first 500 chars): ${stdout.substring(0, 500)}`);
          reject(new Error(`Python inference failed: ${stderr || 'Unknown error'}`));
          return;
        }

        try {
          // YOLO prints progress messages to stdout, so we need to extract only the JSON
          // The JSON should be on the last line (or last few lines if it's multiline)
          const lines = stdout.trim().split('\n');
          let jsonLine = '';
          
          // Find the last line that starts with '{' (our JSON response)
          for (let i = lines.length - 1; i >= 0; i--) {
            const line = lines[i].trim();
            if (line.startsWith('{')) {
              // This might be the start of our JSON - collect from here to the end
              jsonLine = lines.slice(i).join('\n');
              break;
            }
          }
          
          // If we didn't find a line starting with '{', try parsing the whole stdout
          const jsonText = jsonLine || stdout;
          const result = JSON.parse(jsonText);
          
          if (result.error) {
            reject(new Error(result.error));
            return;
          }
          
          // Python returns { detections: [...], annotated_image: "base64..." }
          const detections = Array.isArray(result.detections) ? result.detections : (Array.isArray(result) ? result : []);
          const annotatedImage = result.annotated_image || undefined;
          console.log(`[INFERENCE] Local inference complete: ${detections.length} rooms detected`);
          resolve({ detections, annotated_image: annotatedImage });
        } catch (parseError) {
          console.error(`[INFERENCE] Failed to parse Python output`);
          console.error(`[INFERENCE] First 500 chars of stdout: ${stdout.substring(0, 500)}`);
          console.error(`[INFERENCE] Last 500 chars of stdout: ${stdout.substring(Math.max(0, stdout.length - 500))}`);
          reject(new Error(`Failed to parse inference result: ${parseError}`));
        }
      });

      // Write binary data: model ID length + model ID + image buffer
      pythonProcess.stdin.write(modelIdLength);
      pythonProcess.stdin.write(modelIdBytes);
      pythonProcess.stdin.write(imageBuffer);
      pythonProcess.stdin.end();
    });
  }

  private getMockRooms(): DetectedRoom[] {
    return [
      {
        id: 'room_001',
        bounding_box: [50, 50, 400, 300],
        name_hint: 'Entry Hall'
      },
      {
        id: 'room_002',
        bounding_box: [250, 50, 700, 500],
        name_hint: 'Main Office'
      }
    ];
  }
}

// Singleton instance
export const roomDetector = new RoomDetectionInference();
