export async function detectRooms(file?: File, modelId?: string): Promise<{detections: any[], annotated_image?: string}> {
  if (!file) {
    throw new Error('No file provided');
  }

  // Use Vite proxy in development, or environment variable in production
  const apiUrl = import.meta.env.VITE_API_ENDPOINT || '';
  const fullUrl = apiUrl + '/api/detect';
  
  // Send raw file as multipart/form-data
  const formData = new FormData();
  formData.append('image', file);
  if (modelId) {
    formData.append('model', modelId);
  }
  
  const res = await fetch(fullUrl, {
    method: 'POST',
    body: formData,
  });

  if (!res.ok) throw new Error('detect failed');
  return res.json();
}

