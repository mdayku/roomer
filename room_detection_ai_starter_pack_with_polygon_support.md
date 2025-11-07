# Room Detection AI — Starter Pack

> Starter pack includes: PRD (updated), README, Mermaid architecture diagram, API & schema, and an initial TypeScript codebase scaffold (backend + frontend) with polygon support.

---

## 1) Updated PRD (Polygon-Aware)

### 1.1 Goal
Automatically detect and return room boundaries from architectural blueprints (image or vector-derived raster). Output must support both rectangles and arbitrary polygons.

### 1.2 Context
Existing internal tool extracts room name/number after a user-sketched boundary. We’re automating the boundary step to reduce setup time and boost sales demo impact.

### 1.3 Success Metrics
- **Efficiency:** 10-room plan processed in < 30s
- **Accuracy (MVP):** ≥ 90% of rooms detected on simple plans; ≥ 75% IoU vs. human polygons on moderate plans
- **DX:** One-command local run; clear schemas & decision logs

### 1.4 Scope (MVP)
- Input: uploaded raster blueprint (`.png/.jpg`) or vector->raster preprocessed image
- Output: JSON FeatureCollection with room polygons; each feature has `properties.{id, name_hint?, confidence}`
- Frontend: Upload UI + overlay viewer (SVG) that renders polygons; allow toggle of labels and hovering to show details
- Backend: REST API `/detect` that returns mock detections first; swap to real model later
- Latency: < 30s per plan (mock is instant)

### 1.5 Out-of-Scope (MVP)
- OCR for room names (may come from existing service)
- Vector (PDF) parsing (acceptable to rasterize upstream for MVP)
- Non-room features (stairs, furniture)

### 1.6 Constraints
- Cloud target AWS (future), but MVP runs locally; keep code AWS-ready (12‑factor, clean ports)
- Deterministic mock mode; seedable

---

## 2) Geometry & Data Schemas

### 2.1 GeoJSON-Based Output (recommended)
Use standard GeoJSON for flexibility and tooling support.

```json
{
  "type": "FeatureCollection",
  "image": {
    "width": 4096,
    "height": 3072,
    "normalized": true
  },
  "features": [
    {
      "type": "Feature",
      "properties": {
        "id": "room_001",
        "name_hint": "Entry Hall",
        "confidence": 0.94
      },
      "geometry": {
        "type": "Polygon",
        "coordinates": [
          [ [0.05,0.06], [0.18,0.06], [0.18,0.24], [0.05,0.24], [0.05,0.06] ]
        ]
      }
    }
  ]
}
```
Notes:
- Coordinates normalized to `[0,1]` in both axes (simplifies resolution changes).
- Close the polygon (first point repeats at end).
- Holes supported by additional linear rings in `coordinates`: first ring = outer boundary (CCW), subsequent rings = holes (CW).

### 2.2 Backward-Compatible Bounding Box (optional)
For rectangles, include a convenience bbox in `properties.bbox_norm: [xmin, ymin, xmax, ymax]` (normalized). Frontend will prefer polygon if present.

### 2.3 Input Wall Primitives (for mock/dev)
```json
[
  {"type":"line","start":[100,100],"end":[500,100],"is_load_bearing":false},
  {"type":"line","start":[100,100],"end":[100,400],"is_load_bearing":false}
]
```

---

## 3) API (MVP)

- `POST /api/detect`
  - Body: multipart form with `image` (png/jpg) or JSON `{ imageUrl?: string }`
  - Response: GeoJSON FeatureCollection (as above)

- `GET /api/health` → `{ ok: true }`

---

## 4) Mermaid Diagram

```mermaid
flowchart LR
  U[User: Upload Blueprint] --> FE[React Viewer]
  FE -->|POST /api/detect| API[(Express TS API)]
  API --> DET[Detector Service]
  subgraph DETECTOR
    DET_MOCK[Mock Graph Builder]\n(seedable rooms)
    DET_MODEL[(Future: Model Inference)]
  end
  DET -->|GeoJSON rooms| API --> FE
  FE -->|SVG overlay| CANVAS[Blueprint Canvas]
```

---

## 5) README (Quick Start)

```md
# Room Detection AI (MVP)

Polygon-aware room detection on blueprints (mock first, model later).

## Prereqs
- Node 20+
- pnpm (or npm/yarn)

## Install
pnpm install

## Dev
pnpm dev
# starts both server (http://localhost:4000) and web (http://localhost:5173)

## Build
pnpm build

## Test
pnpm test

## API
POST http://localhost:4000/api/detect → GeoJSON

## Frontend
Upload an image; see SVG overlay polygons with hover details.
```

---

## 6) Initial Codebase (TypeScript)

**Monorepo Structure**
```
room-detection-ai/
  package.json
  pnpm-workspace.yaml
  tsconfig.base.json
  apps/
    server/
      src/
        index.ts
        routes/detect.ts
        services/detector.ts
        utils/geometry.ts
        types/geo.ts
      package.json
      tsconfig.json
    web/
      index.html
      src/
        main.tsx
        App.tsx
        components/Upload.tsx
        components/Overlay.tsx
        lib/api.ts
        types/geo.ts
      vite.config.ts
      package.json
      tsconfig.json
```

### 6.1 `package.json` (root)
```json
{
  "name": "room-detection-ai",
  "private": true,
  "scripts": {
    "dev": "concurrently -k \"pnpm --filter @app/server dev\" \"pnpm --filter @app/web dev\"",
    "build": "pnpm -r build",
    "test": "pnpm -r test"
  },
  "devDependencies": {
    "concurrently": "^9.0.0"
  }
}
```

### 6.2 `apps/server/package.json`
```json
{
  "name": "@app/server",
  "type": "module",
  "scripts": {
    "dev": "tsx watch src/index.ts",
    "build": "tsc -p tsconfig.json",
    "start": "node dist/index.js",
    "test": "vitest"
  },
  "dependencies": {
    "express": "^4.19.2",
    "multer": "^1.4.5-lts.1",
    "zod": "^3.23.8",
    "cors": "^2.8.5"
  },
  "devDependencies": {
    "tsx": "^4.19.0",
    "typescript": "^5.6.3",
    "vitest": "^2.1.3"
  }
}
```

### 6.3 `apps/server/src/types/geo.ts`
```ts
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
```

### 6.4 `apps/server/src/utils/geometry.ts`
```ts
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
```

### 6.5 `apps/server/src/services/detector.ts`
```ts
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
```

### 6.6 `apps/server/src/routes/detect.ts`
```ts
import { Router } from 'express';
import multer from 'multer';
import { mockDetect } from '../services/detector';

const upload = multer({ storage: multer.memoryStorage() });
export const detect = Router();

// POST /api/detect (image optional for mock)
// returns GeoJSON FeatureCollection

detect.post('/detect', upload.single('image'), async (req, res) => {
  const fc = mockDetect();
  res.json(fc);
});

// GET /api/health

detect.get('/health', (_, res) => res.json({ ok: true }));
```

### 6.7 `apps/server/src/index.ts`
```ts
import express from 'express';
import cors from 'cors';
import { detect } from './routes/detect';

const app = express();
app.use(cors());
app.use('/api', detect);

const PORT = process.env.PORT || 4000;
app.listen(PORT, () => console.log(`API listening on :${PORT}`));
```

---

### 6.8 `apps/web/package.json`
```json
{
  "name": "@app/web",
  "private": true,
  "version": "0.1.0",
  "type": "module",
  "scripts": {
    "dev": "vite",
    "build": "vite build",
    "preview": "vite preview",
    "test": "vitest"
  },
  "dependencies": {
    "react": "^18.3.1",
    "react-dom": "^18.3.1"
  },
  "devDependencies": {
    "vite": "^5.4.8",
    "typescript": "^5.6.3",
    "@types/react": "^18.3.5",
    "@types/react-dom": "^18.3.0",
    "vitest": "^2.1.3"
  }
}
```

### 6.9 `apps/web/src/types/geo.ts`
```ts
export type Position = [number, number];
export type LinearRing = Position[];
export type PolygonGeom = { type: 'Polygon'; coordinates: LinearRing[] };
export type Feature = { type:'Feature'; properties: { id:string; name_hint?:string; confidence?:number; bbox_norm?:[number,number,number,number] }; geometry: PolygonGeom };
export type FeatureCollection = { type:'FeatureCollection'; image?: { width?:number; height?:number; normalized?:boolean }; features: Feature[] };
```

### 6.10 `apps/web/src/lib/api.ts`
```ts
import type { FeatureCollection } from '../types/geo';

export async function detectRooms(file?: File): Promise<FeatureCollection> {
  const form = new FormData();
  if (file) form.append('image', file);
  const res = await fetch('http://localhost:4000/api/detect', { method:'POST', body: form });
  if (!res.ok) throw new Error('detect failed');
  return res.json();
}
```

### 6.11 `apps/web/src/components/Overlay.tsx`
```tsx
import React from 'react';
import type { FeatureCollection, Position } from '../types/geo';

function pathFromRing(ring: Position[]): string {
  return ring.map((p,i)=>`${i?'L':'M'} ${p[0]*1000} ${p[1]*1000}`).join(' ') + ' Z';
}

export const Overlay: React.FC<{ fc: FeatureCollection|null }>=({ fc })=>{
  if (!fc) return null;
  return (
    <svg viewBox="0 0 1000 1000" style={{position:'absolute', inset:0, pointerEvents:'none'}}>
      {fc.features.map((f)=> (
        <g key={f.properties.id}>
          {f.geometry.coordinates.map((ring, idx)=> (
            <path key={idx} d={pathFromRing(ring)} fill="none" strokeWidth={2} strokeDasharray="6 4" strokeOpacity={0.9} />
          ))}
          {/* label */}
          <text x={(f.properties.bbox_norm?.[0]??0.5)*1000} y={(f.properties.bbox_norm?.[1]??0.5)*1000 - 6} fontSize={18}>
            {f.properties.name_hint ?? f.properties.id}
          </text>
        </g>
      ))}
    </svg>
  );
};
```

### 6.12 `apps/web/src/components/Upload.tsx`
```tsx
import React, { useRef, useState } from 'react';
import { detectRooms } from '../lib/api';
import type { FeatureCollection } from '../types/geo';

export const Upload: React.FC<{ onResult:(fc:FeatureCollection)=>void }>=({ onResult })=>{
  const [busy,setBusy] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  return (
    <div style={{display:'flex',gap:12}}>
      <input ref={inputRef} type="file" accept="image/*" />
      <button disabled={busy} onClick={async()=>{
        try{
          setBusy(true);
          const file = inputRef.current?.files?.[0] ?? undefined;
          const fc = await detectRooms(file);
          onResult(fc);
        } finally {
          setBusy(false);
        }
      }}>Detect Rooms</button>
    </div>
  );
};
```

### 6.13 `apps/web/src/App.tsx`
```tsx
import React, { useState } from 'react';
import { Upload } from './components/Upload';
import { Overlay } from './components/Overlay';
import type { FeatureCollection } from './types/geo';

export default function App(){
  const [imgUrl,setImgUrl] = useState<string|null>(null);
  const [fc,setFc] = useState<FeatureCollection|null>(null);

  return (
    <div style={{padding:16}}>
      <h2>Room Detection AI (Polygon MVP)</h2>
      <Upload onResult={(f)=>setFc(f)} />
      <div style={{ position:'relative', width:800, height:600, marginTop:12, background:'#fafafa', border:'1px solid #ddd', display:'grid', placeItems:'center' }}>
        {imgUrl ? <img src={imgUrl} style={{maxWidth:'100%',maxHeight:'100%'}}/> : <em>Upload a floor plan image (optional)</em>}
        <Overlay fc={fc} />
      </div>
    </div>
  );
}
```

### 6.14 `apps/web/src/main.tsx`
```tsx
import React from 'react';
import { createRoot } from 'react-dom/client';
import App from './App';

createRoot(document.getElementById('root')!).render(<App/>);
```

### 6.15 `apps/web/index.html`
```html
<!doctype html>
<html>
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>Room Detection AI</title>
  </head>
  <body>
    <div id="root"></div>
    <script type="module" src="/src/main.tsx"></script>
  </body>
</html>
```

---

## 7) Next Steps

1) **Ground-Truth Format**: Store hand-labeled rooms as GeoJSON; define naming convention and class list (room vs corridor).
2) **Vector Preprocessing**: If source is PDF, rasterize at appropriate DPI (e.g., 300) to simplify MVP.
3) **Classical First**: Wall-line detection (Canny + Hough) → polygonization (contour -> polygon). Compare with learned model later.
4) **Validation**: IoU metrics vs human polygons; track latency and coverage.
5) **Name Hints**: Optional integration with existing room-name OCR; attach to `properties.name_hint`.
6) **AWS Hardening**: Containerize; add S3 upload + Lambda/SageMaker inference skeleton.

---

**End of Starter Pack**

