import type { FeatureCollection } from '../types/geo';

export async function detectRooms(file?: File, modelId?: string): Promise<FeatureCollection> {
  const form = new FormData();
  if (file) form.append('image', file);
  if (modelId) form.append('model', modelId);

  // Use deployed API endpoint in production, localhost for development
  const apiUrl = import.meta.env.VITE_API_URL || import.meta.env.REACT_APP_API_ENDPOINT || 'http://localhost:4000';
  const res = await fetch(`${apiUrl}/api/detect`, { method:'POST', body: form });

  if (!res.ok) throw new Error('detect failed');
  return res.json();
}

