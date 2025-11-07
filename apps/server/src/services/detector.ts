import type { FeatureCollection, Feature } from '../types/geo';
import { rectToPolygon } from '../utils/geometry';

export function mockDetect(seed = 42): FeatureCollection {
  // Very simple seeded mock rooms
  const rooms: Feature[] = [
    {
      type: 'Feature',
      properties: { id: 'room_001', name_hint: 'Entry Hall', confidence: 0.95, bbox_norm: [0.05,0.06,0.18,0.24] },
      geometry: rectToPolygon(0.05,0.06,0.18,0.24)
    },
    {
      type: 'Feature',
      properties: { id: 'room_002', name_hint: 'Main Office', confidence: 0.92, bbox_norm: [0.22,0.06,0.68,0.45] },
      geometry: {
        type: 'Polygon',
        coordinates: [[
          [0.22,0.06],[0.68,0.06],[0.68,0.28],[0.54,0.28],[0.54,0.45],[0.22,0.45],[0.22,0.06]
        ]]
      }
    }
  ];
  return { type:'FeatureCollection', image:{ normalized:true }, features: rooms };
}

