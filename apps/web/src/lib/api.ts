import type { FeatureCollection } from '../types/geo';

export async function detectRooms(file?: File): Promise<FeatureCollection> {
  const form = new FormData();
  if (file) form.append('image', file);
  const res = await fetch('http://localhost:4000/api/detect', { method:'POST', body: form });
  if (!res.ok) throw new Error('detect failed');
  return res.json();
}

