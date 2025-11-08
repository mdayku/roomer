import type { FeatureCollection } from '../types/geo';

export async function detectRooms(file?: File, modelId?: string): Promise<FeatureCollection> {
  const form = new FormData();
  if (file) form.append('image', file);
  if (modelId) form.append('model', modelId);

  // Use deployed API endpoint in production, localhost for development
  const apiUrl = process.env.REACT_APP_API_ENDPOINT || 'https://qhhfso7u0h.execute-api.us-east-1.amazonaws.com/prod';
  const fullUrl = apiUrl + '/api/detect';
  const res = await fetch(fullUrl, { method:'POST', body: form });

  if (!res.ok) throw new Error('detect failed');
  return res.json();
}

