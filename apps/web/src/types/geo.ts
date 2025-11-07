export type Position = [number, number];
export type LinearRing = Position[];
export type PolygonGeom = { type: 'Polygon'; coordinates: LinearRing[] };
export type Feature = { type:'Feature'; properties: { id:string; name_hint?:string; confidence?:number; bbox_norm?:[number,number,number,number] }; geometry: PolygonGeom };
export type FeatureCollection = { type:'FeatureCollection'; image?: { width?:number; height?:number; normalized?:boolean }; features: Feature[] };

