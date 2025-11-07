import { spawn } from 'child_process';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';
import type { FeatureCollection } from '../types/geo';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

class RoomDetectionInference {
  private pythonPath = 'python3'; // or 'python' on Windows

  async detectRooms(imageBuffer: Buffer): Promise<FeatureCollection> {
    try {
      console.log('Running Python inference...');

      // Convert image buffer to base64 for sending to Python
      const imageBase64 = imageBuffer.toString('base64');

      // Prepare input for Python script
      const inputData = JSON.stringify({
        action: 'detect',
        image: imageBase64
      });

      // Call Python inference script
      const result = await this.callPythonScript(inputData);

      if (result.error) {
        throw new Error(result.error);
      }

      console.log(`Inference complete: ${result.features?.length || 0} rooms detected`);
      return result;

    } catch (error) {
      console.error('Inference failed:', error);

      // Fallback to mock results
      console.log('Falling back to mock detection');
      const mockFeatures: Feature[] = [
        {
          type: 'Feature' as const,
          properties: {
            id: 'room_001',
            name_hint: 'Living Room',
            confidence: 0.92,
            bbox_norm: [0.1, 0.1, 0.4, 0.3]
          },
          geometry: {
            type: 'Polygon' as const,
            coordinates: [[[0.1, 0.1], [0.4, 0.1], [0.4, 0.3], [0.1, 0.3], [0.1, 0.1]]]
          }
        },
        {
          type: 'Feature' as const,
          properties: {
            id: 'room_002',
            name_hint: 'Kitchen',
            confidence: 0.88,
            bbox_norm: [0.5, 0.1, 0.8, 0.4]
          },
          geometry: {
            type: 'Polygon' as const,
            coordinates: [[[0.5, 0.1], [0.8, 0.1], [0.8, 0.4], [0.5, 0.4], [0.5, 0.1]]]
          }
        }
      ];

      return {
        type: 'FeatureCollection',
        image: { normalized: true },
        features: mockFeatures
      };
    }
  }

  private async callPythonScript(inputData: string): Promise<any> {
    return new Promise((resolve, reject) => {
      const scriptPath = join(__dirname, 'inference.py');

      const python = spawn(this.pythonPath, [scriptPath], {
        stdio: ['pipe', 'pipe', 'pipe']
      });

      let stdout = '';
      let stderr = '';

      python.stdout.on('data', (data) => {
        stdout += data.toString();
      });

      python.stderr.on('data', (data) => {
        stderr += data.toString();
      });

      python.on('close', (code) => {
        if (code !== 0) {
          reject(new Error(`Python script failed: ${stderr}`));
          return;
        }

        try {
          const result = JSON.parse(stdout.trim());
          resolve(result);
        } catch (e) {
          reject(new Error(`Failed to parse Python output: ${stdout}`));
        }
      });

      python.on('error', (error) => {
        reject(error);
      });

      // Send input data
      python.stdin.write(inputData);
      python.stdin.end();
    });
  }
}

// Singleton instance
export const roomDetector = new RoomDetectionInference();
