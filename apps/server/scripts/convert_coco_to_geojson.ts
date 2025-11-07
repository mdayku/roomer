/*
  COCO → GeoJSON (Polygon) converter for room annotations

  Supports:
  - COCO polygon segmentations (arrays of [x1,y1,...]) → rings
  - COCO RLE segmentations (uncompressed {counts:[], size:[]}) and compressed RLE ({counts:string, size:[]})
    → decodes to a binary mask → extracts polygon rings with marching squares → simplifies + normalizes.

  Output:
  - One GeoJSON FeatureCollection per image (rooms only), coordinates normalized to [0,1]

  Usage:
    # Install dependencies first
    pnpm add -w marchingsquares
    pnpm --filter @app/server add -D tsx

    # Convert CubiCasa5K COCO annotations (category_id 2 = "room")
    # Windows PowerShell (single line):
    pnpm --filter @app/server tsx scripts/convert_coco_to_geojson.ts --ann ../../cubicasa5k_coco/train_coco_pt.json --out ../../cubicasa5k_geojson --room-cat-ids 2 --min-area 20
    
    # Unix/Linux/Mac (multi-line with backslash):
    pnpm --filter @app/server tsx scripts/convert_coco_to_geojson.ts \
      --ann ../../cubicasa5k_coco/train_coco_pt.json \
      --out ../../cubicasa5k_geojson \
      --room-cat-ids 2 \
      --min-area 20

    # Or auto-detect room category from COCO file (Windows):
    pnpm --filter @app/server tsx scripts/convert_coco_to_geojson.ts --ann ../../cubicasa5k_coco/train_coco_pt.json --out ../../cubicasa5k_geojson --auto-room-cat --min-area 20
*/

import fs from 'node:fs';
import path from 'node:path';
import yargs from 'yargs';
import { hideBin } from 'yargs/helpers';
// @ts-ignore - types may be missing
import { isoLines } from 'marchingsquares';
import type { Position, Feature, FeatureCollection, FeatureProperties } from '../src/types/geo.js';

// -------- COCO Types --------
interface CocoImage { id:number; file_name:string; width:number; height:number; }
interface CocoAnn {
  id:number; image_id:number; category_id:number;
  segmentation: number[] | number[][] | { counts:number[]|string; size:[number,number] };
  area?: number; iscrowd?: 0|1; bbox?: [number,number,number,number];
}
interface Coco { images:CocoImage[]; annotations:CocoAnn[]; categories: {id:number; name:string}[] }

// -------- CLI --------
const argv = yargs(hideBin(process.argv))
  .option('ann', { type:'string', demandOption:true, desc:'Path to COCO annotations JSON' })
  .option('out', { type:'string', demandOption:true, desc:'Output folder for GeoJSON files' })
  .option('room-cat-ids', { type:'array', desc:'Category IDs that represent rooms (e.g., 2 for "room")' })
  .option('auto-room-cat', { type:'boolean', desc:'Auto-detect room category by name (case-insensitive match for "room")', default: false })
  .option('min-area', { type:'number', desc:'Minimum polygon area in pixels to keep (before normalization)', default: 0 })
  .parseSync();

const ANN_PATH = path.resolve(argv.ann);
const OUTPUT_DIR = path.resolve(argv.out);
const MIN_AREA = Number(argv['min-area'] || 0);
const AUTO_ROOM_CAT = Boolean(argv['auto-room-cat']);

if (!fs.existsSync(ANN_PATH)) {
  console.error('Error: Annotation file not found:', ANN_PATH);
  process.exit(1);
}

fs.mkdirSync(OUTPUT_DIR, { recursive: true });

// -------- Helpers --------
function closeRing(ring: Position[]): Position[] {
  if (ring.length<3) return ring;
  const [fx,fy] = ring[0];
  const [lx,ly] = ring[ring.length-1];
  if (fx!==lx || fy!==ly) return [...ring, ring[0]];
  return ring;
}
function areaPx(ring: Position[]): number {
  let a = 0; for (let i=0;i<ring.length-1;i++){ const [x1,y1]=ring[i]; const [x2,y2]=ring[i+1]; a += (x1*y2 - x2*y1);} return Math.abs(a)/2;
}
function normalize(ring: Position[], w:number,h:number): Position[]{ return ring.map(([x,y])=>[x/w,y/h]); }
function simplify(ring: Position[], eps = 0.5): Position[] { // RDP in-place simple
  // Very small epsilon by default; replace with robust lib if needed
  if (ring.length<=3) return ring;
  const sq = (x:number)=>x*x;
  function dist2(p:Position,a:Position,b:Position){
    const [x,y]=p,[x1,y1]=a,[x2,y2]=b; const dx=x2-x1, dy=y2-y1; if (dx===0&&dy===0) return sq(x-x1)+sq(y-y1);
    const t = ((x-x1)*dx+(y-y1)*dy)/(dx*dx+dy*dy); const xx=x1+t*dx, yy=y1+t*dy; return sq(x-xx)+sq(y-yy);
  }
  function rdp(pts:Position[], s:number, e:number, out:number[]){
    let maxD = 0, idx = s+1; for (let i=s+1;i<e;i++){ const d=dist2(pts[i], pts[s], pts[e]); if (d>maxD){ maxD=d; idx=i; } }
    if (Math.sqrt(maxD) > eps){ rdp(pts,s,idx,out); rdp(pts,idx,e,out); } else { out.push(s,e); }
  }
  const last = ring.length-1; const keep:number[]=[]; rdp(ring,0,last,keep);
  const uniq = Array.from(new Set(keep)).sort((a,b)=>a-b).map(i=>ring[i]);
  return closeRing(uniq);
}

// COCO RLE decode (compressed or uncompressed) → Uint8Array mask of size w*h
function rleDecode(rle: {counts:number[]|string; size:[number,number]}): Uint8Array {
  const [h,w] = rle.size; // COCO stores [h,w]
  const a = new Uint8Array(w*h);
  let counts:number[];
  if (Array.isArray(rle.counts)) counts = rle.counts as number[];
  else counts = decodeCompressedRLE(rle.counts as string);
  let idx = 0; let val = 0; let p = 0;
  for (const c of counts){ for (let i=0;i<c;i++){ a[p++] = val; } val = val ? 0 : 1; idx++; }
  return a;
}

// Decode COCO compressed RLE string → number[] counts
// See: https://cocodataset.org/#format-data (ASCII run-length encoding)
function decodeCompressedRLE(s: string): number[] {
  const counts:number[] = []; let p=0; let m=0; let val=0;
  while (p < s.length){ let x=0; let k=0; let more=1;
    while (more){ const c = s.charCodeAt(p++) - 48; x |= (c & 0x1f) << (5*k); more = c & 0x20; k++; if (!more && (c & 0x10)) x |= -1 << (5*k); }
    if (m && x===0) val += 31; else val += x; counts.push(val); m = 1; val = 0;
  }
  // The above is a compact variant; for robustness, consider proven libraries in production.
  return counts;
}

function maskToPolygons(mask: Uint8Array, w:number,h:number): Position[][] {
  // marching squares expects a 2D array; build row-major grid of 0/1
  const grid:number[][] = new Array(h);
  for (let y=0;y<h;y++){ const row:number[] = new Array(w); for (let x=0;x<w;x++){ row[x] = mask[y*w + x] ? 1 : 0; } grid[y] = row; }
  const iso = 0.5; // threshold
  // isoLines returns an array of line paths; each path is an array of [x,y] coordinates
  const result = isoLines(grid, iso, { polygons: true, linearRing: true });
  const out: Position[][] = [];
  // result is an array of paths, each path is an array of [x,y] coordinates
  if (Array.isArray(result)) {
    for (const path of result) {
      if (Array.isArray(path) && path.length > 0) {
        const ring: Position[] = path.map((pt: any) => {
          // Handle both [x,y] and {x, y} formats
          if (Array.isArray(pt)) return [pt[0], pt[1]] as Position;
          if (pt && typeof pt === 'object' && 'x' in pt && 'y' in pt) return [pt.x, pt.y] as Position;
          return [0, 0] as Position;
        }).filter((p: Position) => p[0] !== 0 || p[1] !== 0);
        if (ring.length >= 3) {
          out.push(closeRing(ring));
        }
      }
    }
  }
  return out;
}

function cocoPolyToRing(seg: number[]): Position[] {
  const ring: Position[] = [];
  for (let i=0;i<seg.length;i+=2){ ring.push([seg[i], seg[i+1]]); }
  return closeRing(ring);
}

// -------- Main --------
console.log('Reading COCO annotations from:', ANN_PATH);
let coco: Coco;
try {
  coco = JSON.parse(fs.readFileSync(ANN_PATH, 'utf8'));
} catch (err) {
  console.error('Error reading annotation file:', err);
  process.exit(1);
}

console.log(`Found ${coco.images.length} images, ${coco.annotations.length} annotations, ${coco.categories.length} categories`);

// Determine room category IDs
let ROOM_CAT_IDS: number[] = [];
if (AUTO_ROOM_CAT) {
  // Auto-detect room category by name
  for (const cat of coco.categories) {
    if (cat.name.toLowerCase().includes('room')) {
      ROOM_CAT_IDS.push(cat.id);
      console.log(`Auto-detected room category: ${cat.name} (id: ${cat.id})`);
    }
  }
  if (ROOM_CAT_IDS.length === 0) {
    console.warn('Warning: No room category found with --auto-room-cat. Available categories:', coco.categories.map(c => `${c.name} (${c.id})`).join(', '));
  }
} else {
  ROOM_CAT_IDS = (argv['room-cat-ids'] as number[] | string[] | undefined)?.map(Number) || [];
  if (ROOM_CAT_IDS.length === 0) {
    console.warn('Warning: No room category IDs specified. Use --room-cat-ids or --auto-room-cat');
  }
}

// Build annotation index by image
const annsByImage = new Map<number, CocoAnn[]>();
for (const ann of coco.annotations) {
  if (ROOM_CAT_IDS.length && !ROOM_CAT_IDS.includes(Number(ann.category_id))) continue;
  const list = annsByImage.get(ann.image_id) || [];
  list.push(ann);
  annsByImage.set(ann.image_id, list);
}

// Build category name map
const catMap = new Map<number, string>();
for (const cat of coco.categories) {
  catMap.set(cat.id, cat.name);
}

let processed = 0;
let totalRooms = 0;

// Process all images (remove debug limit)
for (const img of coco.images) {
  const anns = annsByImage.get(img.id) || [];
  const features: Feature[] = [];
  let idx = 0;
  
  for (const ann of anns) {
    try {
      let ring: Position[] | null = null;
      
    // case 1: polygon(s)
      if (ann.segmentation && Array.isArray(ann.segmentation)) {
        const segs = Array.isArray(ann.segmentation[0]) 
          ? (ann.segmentation as number[][]) 
          : [ann.segmentation as number[]];
        for (const seg of segs) {
          if (seg.length < 6) continue; // Need at least 3 points (x,y pairs)
          ring = cocoPolyToRing(seg);
          break; // Use first valid polygon
      }
    }
    // case 2: RLE
      else if (ann.segmentation && typeof ann.segmentation === 'object' && 'size' in ann.segmentation) {
      const rle = ann.segmentation as {counts:number[]|string; size:[number,number]};
        try {
      const mask = rleDecode(rle);
      const polys = maskToPolygons(mask, rle.size[1], rle.size[0]); // note size = [h,w]
          if (polys.length > 0) {
            ring = polys[0]; // Use first polygon
          }
        } catch (err) {
          console.warn(`Warning: Failed to decode RLE for annotation ${ann.id}:`, err);
        }
      }
      // case 3: bbox only (COCO format: [x, y, width, height])
      else if (ann.bbox && Array.isArray(ann.bbox) && ann.bbox.length === 4) {
        const [x, y, w, h] = ann.bbox;
        ring = closeRing([
          [x, y],
          [x + w, y],
          [x + w, y + h],
          [x, y + h]
        ]);
      }
      
      // Process the ring if we have one
      if (ring && ring.length >= 3) {
        if (MIN_AREA > 0 && areaPx(ring) < MIN_AREA) continue;
        
        const catName = catMap.get(ann.category_id) || 'room';
        const ringN = normalize(ring, img.width, img.height);
        
        // Calculate bbox_norm for convenience
        const xs = ring.map(p => p[0]);
        const ys = ring.map(p => p[1]);
        const bbox_norm: [number, number, number, number] = [
          Math.min(...xs) / img.width,
          Math.min(...ys) / img.height,
          Math.max(...xs) / img.width,
          Math.max(...ys) / img.height
        ];
        
        const props: FeatureProperties = {
          id: `room_${String(++idx).padStart(4, '0')}`,
          name_hint: catName !== 'room' ? catName : undefined,
          confidence: 1.0, // Ground truth data
          bbox_norm
        };
        
        features.push({
          type: 'Feature',
          properties: props,
          geometry: {
            type: 'Polygon',
            coordinates: [ringN]
          }
        });
      }
    } catch (err) {
      console.warn(`Warning: Failed to process annotation ${ann.id} for image ${img.id}:`, err);
      }
    }
  
  const fc: FeatureCollection = {
    type: 'FeatureCollection',
    image: {
      width: img.width,
      height: img.height,
      normalized: true
    },
    features
  };
  
  // Create unique filename using image ID to avoid collisions
  const baseName = path.basename(img.file_name, path.extname(img.file_name));
  const uniqueName = `${baseName}_${img.id}.geo.json`;
  const outPath = path.join(OUTPUT_DIR, uniqueName);

  fs.writeFileSync(outPath, JSON.stringify(fc, null, 2));
  processed++;
  totalRooms += features.length;

  if (processed % 100 === 0 || processed === coco.images.length) {
    console.log(`Processed ${processed}/${coco.images.length} images...`);
  }
}

console.log(`\nDone! Processed ${processed} images, extracted ${totalRooms} room polygons`);
console.log(`Output directory: ${OUTPUT_DIR}`);
