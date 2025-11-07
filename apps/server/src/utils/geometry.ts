import type { LinearRing, PolygonGeom } from '../types/geo';

export const closeRing = (ring: LinearRing): LinearRing => {
  if (ring.length < 3) return ring;
  const first = ring[0];
  const last = ring[ring.length - 1];
  if (first[0] !== last[0] || first[1] !== last[1]) return [...ring, first];
  return ring;
};

export const rectToPolygon = (xmin:number,ymin:number,xmax:number,ymax:number): PolygonGeom => ({
  type: 'Polygon',
  coordinates: [
    closeRing([
      [xmin, ymin],[xmax, ymin],[xmax, ymax],[xmin, ymax]
    ])
  ]
});

export const area = (ring: LinearRing): number => {
  let a = 0;
  for (let i = 0; i < ring.length - 1; i++) {
    const [x1,y1] = ring[i];
    const [x2,y2] = ring[i+1];
    a += (x1*y2 - x2*y1);
  }
  return Math.abs(a) / 2;
};

