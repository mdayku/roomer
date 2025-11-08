import type { FeatureCollection } from '../types/geo';

export async function detectRooms(file?: File, modelId?: string): Promise<FeatureCollection> {
  // Convert image to base64 if provided
  let imageBase64: string | undefined;
  if (file) {
    imageBase64 = await new Promise<string>((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => {
        const result = reader.result as string;
        // Remove data URL prefix (e.g., "data:image/png;base64,")
        const base64 = result.includes(',') ? result.split(',')[1] : result;
        resolve(base64);
      };
      reader.onerror = reject;
      reader.readAsDataURL(file);
    });
  }

  // Use deployed API endpoint in production, localhost for development
  const apiUrl = process.env.REACT_APP_API_ENDPOINT || 'https://qhhfso7u0h.execute-api.us-east-1.amazonaws.com/prod';
  const fullUrl = apiUrl + '/api/detect';
  
  const res = await fetch(fullUrl, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      image: imageBase64,
      model: modelId,
    }),
  });

  if (!res.ok) throw new Error('detect failed');
  return res.json();
}

