export type Position = [number, number]; // normalized [x,y] in [0,1]
export type LinearRing = Position[]; // closed (first==last)

export interface FeatureProperties {
  id: string;
  name_hint?: string;
  confidence?: number; // 0..1
  bbox_norm?: [number, number, number, number]; // optional convenience
}

export interface PolygonGeom {
  type: 'Polygon';
  coordinates: LinearRing[]; // outer ring + holes
}

export interface Feature {
  type: 'Feature';
  properties: FeatureProperties;
  geometry: PolygonGeom; // MVP supports Polygon
}

export interface FeatureCollection {
  type: 'FeatureCollection';
  image?: { width?: number; height?: number; normalized?: boolean };
  features: Feature[];
}

