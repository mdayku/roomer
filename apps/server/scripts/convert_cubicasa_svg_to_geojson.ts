/*
  CubiCasa SVG → GeoJSON (Polygon) converter
  - Supports <polygon points>, <polyline points> (closed), and <path d> with M/L/Z commands
  - Normalizes coordinates to [0,1] using SVG viewBox when available; otherwise width/height
  - Filters polygons by class names (e.g., "room", "Area"), or include all if none provided

  Usage:
    # install deps in the monorepo root or server package
    pnpm add -w fast-xml-parser glob yargs
    pnpm --filter @app/server add -D tsx

    # run
    pnpm --filter @app/server tsx scripts/convert_cubicasa_svg_to_geojson.ts \
      --in ../../cubicasa5k/cubicasa5k \
      --out ../../cubicasa5k_svg_geojson \
      --room-classes room Room Area Space
*/

import fs from 'node:fs';
import path from 'node:path';
import { glob } from 'glob';
import { XMLParser } from 'fast-xml-parser';
import yargs from 'yargs';
import { hideBin } from 'yargs/helpers';
import type { Position, Feature, FeatureCollection } from '../src/types/geo.js';

const argv = yargs(hideBin(process.argv))
  .option('in', { type:'string', demandOption:true, desc:'Input folder with SVG files' })
  .option('out', { type:'string', demandOption:true, desc:'Output folder for GeoJSON files' })
  .option('room-classes', { type:'array', desc:'Class names to keep as rooms (case-insensitive). If omitted, include all polygons.' })
  .parseSync();

const INPUT_DIR = path.resolve(argv.in);
const OUTPUT_DIR = path.resolve(argv.out);
const ROOM_CLASSES = (argv['room-classes'] as string[] | undefined)?.map(s=>s.toLowerCase());

fs.mkdirSync(OUTPUT_DIR, { recursive: true });

const parser = new XMLParser({
  ignoreAttributes: false,
  attributeNamePrefix: '@_',
  allowBooleanAttributes: true
});

function parsePoints(pointsAttr: string): Position[] {
  // accepts "x1,y1 x2,y2 ..." or "x1,y1\nx2,y2"
  return pointsAttr.trim().split(/\s+/).map(token => {
    const [x,y] = token.split(',').map(Number);
    return [x,y];
  });
}

function parsePath(d: string): Position[] {
  // Minimal M/L/Z parser; ignores curves (C/Q) — assumes rooms are rectilinear/polylines
  const tokens = d.match(/[MLZmlz]|-?\d*\.?\d+(?:e[-+]?\d+)?/g) || [];
  let i = 0; let cmd = '';
  let cursor: Position = [0,0];
  const out: Position[] = [];
  const push = (p: Position)=>{ out.push(p); cursor = p; };

  while (i < tokens.length) {
    const t = tokens[i++];
    if (/^[MLZmlz]$/.test(t)) { cmd = t; if (cmd==='Z'||cmd==='z') break; continue; }
    // otherwise t is a number; rewind one token to consume as coords
    i--;
    if (cmd==='M' || cmd==='L') {
      const x = Number(tokens[i++]);
      const y = Number(tokens[i++]);
      push([x,y]);
    } else if (cmd==='m' || cmd==='l') {
      const dx = Number(tokens[i++]);
      const dy = Number(tokens[i++]);
      push([cursor[0]+dx, cursor[1]+dy]);
    } else {
      // unsupported command; skip a number to avoid infinite loop
      i++;
    }
  }
  return out;
}

function closeRing(ring: Position[]): Position[] {
  if (ring.length<3) return ring;
  const [fx,fy] = ring[0];
  const [lx,ly] = ring[ring.length-1];
  if (fx!==lx || fy!==ly) return [...ring, ring[0]];
  return ring;
}

function normalize(ring: Position[], width:number, height:number): Position[] {
  return ring.map(([x,y]) => [x/width, y/height]);
}

function hasRoomClass(attrs: Record<string, any>): boolean {
  if (!ROOM_CLASSES || ROOM_CLASSES.length===0) return true; // include all
  // Try different attribute access patterns including inherited class
  const cls = String(
    attrs.inheritedClass || // inherited from parent
    attrs['@_class'] ||     // prefixed attribute
    attrs.class ||          // direct attribute
    (attrs as any)['className'] || // alternative name
    ''                      // fallback
  ).toLowerCase();

  const hasMatch = ROOM_CLASSES.some(c => cls.split(/\s+/).includes(c.toLowerCase()));
  if (hasMatch) {
    console.log(`Found room class: "${cls}" matches one of: [${ROOM_CLASSES.join(', ')}]`);
  }
  return hasMatch;
}

function toFeatures(svg: any): { features: Feature[]; vw:number; vh:number } {
  // Read viewBox or width/height
  const root = svg.svg;
  const viewBox = (root?.viewBox || '').toString().trim();
  let vw = Number(root?.width) || 0;
  let vh = Number(root?.height) || 0;
  if (viewBox) {
    const parts = viewBox.split(/\s+/).map(Number);
    if (parts.length===4) { vw = parts[2]; vh = parts[3]; }
  }
  if (!vw || !vh) { vw = 1000; vh = 1000; } // fallback

  console.log(`SVG dimensions: ${vw}x${vh}`);

  const elements: Array<{ tag:string; attrs:any }> = [];
  function walk(node: any, parentClass = '') {
    if (!node || typeof node!== 'object') return;
    for (const [key, val] of Object.entries(node)) {
      if (key === 'polygon' || key === 'polyline' || key === 'path') {
        const list = Array.isArray(val) ? val : [val];
        for (const el of list) {
          // Inherit class from parent if element doesn't have its own
          const elementClass = el['@_class'] || parentClass;
          elements.push({
            tag: key,
            attrs: { ...el, inheritedClass: elementClass }
          });
        }
      } else if (typeof val === 'object') {
        // Pass down the class from this level to children
        const currentClass = val['@_class'] || parentClass;
        walk(val, currentClass);
      }
    }
  }
  walk(root);

  console.log(`Found ${elements.length} elements:`, elements.slice(0, 10).map(el => `${el.tag} with attrs: ${JSON.stringify(el.attrs)}`));

  const feats: Feature[] = [];
  let idx = 0;
  for (const el of elements) {
    const a = el.attrs || {};
    console.log(`Checking element: ${el.tag}, class="${a.class || 'none'}"`);
    if (!hasRoomClass(a)) {
      console.log(`Skipping element: no room class match`);
      continue;
    }

    console.log(`Processing room element: ${el.tag}`);
    let ring: Position[] = [];
    if (el.tag==='polygon' && (a['@_points'] || a.points)) {
      const pointsStr = String(a['@_points'] || a.points);
      ring = parsePoints(pointsStr);
      console.log(`Parsed polygon with ${ring.length} points from: ${pointsStr.substring(0, 50)}...`);
    }
    else if (el.tag==='polyline' && (a['@_points'] || a.points)) {
      const pointsStr = String(a['@_points'] || a.points);
      ring = closeRing(parsePoints(pointsStr));
      console.log(`Parsed polyline with ${ring.length} points`);
    }
    else if (el.tag==='path' && (a['@_d'] || a.d)) {
      const pathStr = String(a['@_d'] || a.d);
      ring = closeRing(parsePath(pathStr));
      console.log(`Parsed path with ${ring.length} points`);
    }

    if (ring.length >= 3) {
      const ringN = normalize(ring, vw, vh);
      feats.push({
        type:'Feature',
        properties:{ id: `room_${String(++idx).padStart(4,'0')}` },
        geometry:{ type:'Polygon', coordinates:[ringN] }
      });
      console.log(`Added room feature ${idx}`);
    } else {
      console.log(`Skipping: ring has only ${ring.length} points`);
    }
  }
  console.log(`Total features extracted: ${feats.length}`);
  return { features: feats, vw, vh };
}

async function run() {
  console.log('Looking for SVG files in:', INPUT_DIR);
  const files = await glob('**/*.svg', { cwd: INPUT_DIR, absolute: true, nodir: true });
  console.log('Found', files.length, 'SVG files:', files.map(f => path.basename(f)));
  if (files.length===0) {
    console.error('No SVG files found in', INPUT_DIR);
    process.exit(1);
  }
  let totalProcessed = 0;
  let totalRooms = 0;

  for (const f of files) {
    const raw = fs.readFileSync(f, 'utf8');
    const svg = parser.parse(raw);
    const { features, vw, vh } = toFeatures(svg);

    const fc: FeatureCollection = { type:'FeatureCollection', image:{ width: vw, height: vh, normalized: true }, features };
    const outPath = path.join(OUTPUT_DIR, path.basename(f).replace(/\.svg$/i, '.geo.json'));
    fs.writeFileSync(outPath, JSON.stringify(fc, null, 2));

    totalProcessed++;
    totalRooms += features.length;

    if (totalProcessed % 100 === 0 || totalProcessed === files.length) {
      console.log(`Processed ${totalProcessed}/${files.length} SVG files...`);
    }
  }

  console.log(`\nDone! Processed ${totalProcessed} SVG files, extracted ${totalRooms} room polygons`);
  console.log(`Output directory: ${OUTPUT_DIR}`);
}

run().catch(err=>{ console.error(err); process.exit(1); });

