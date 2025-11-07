/*
  Convert GeoJSON dataset to COCO format for ML training

  Preserves polygon coordinates while making it ML-friendly.
  COCO format supports polygon segmentation (not just bboxes).

  Usage:
    pnpm --filter @app/server tsx scripts/convert_geojson_to_coco.ts \
      --input-dir room_detection_dataset_clean \
      --output-dir room_detection_dataset_coco
*/

import fs from 'node:fs';
import path from 'node:path';
import yargs from 'yargs';
import { hideBin } from 'yargs/helpers';
import type { FeatureCollection, Position } from '../src/types/geo.js';

const argv = yargs(hideBin(process.argv))
  .option('input-dir', { type:'string', demandOption:true, desc:'Input GeoJSON dataset directory' })
  .option('output-dir', { type:'string', demandOption:true, desc:'Output COCO dataset directory' })
  .parseSync();

const INPUT_DIR = path.resolve(argv['input-dir']);
const OUTPUT_DIR = path.resolve(argv['output-dir']);

// -------- COCO Types --------
interface CocoImage {
  id: number;
  file_name: string;
  width: number;
  height: number;
}

interface CocoAnnotation {
  id: number;
  image_id: number;
  category_id: number;
  segmentation: number[][]; // polygon coordinates: [[x1,y1,x2,y2,...], [x1,y1,x2,y2,...]]
  area: number;
  bbox: [number, number, number, number]; // [x, y, width, height]
  iscrowd: 0 | 1;
}

interface CocoCategory {
  id: number;
  name: string;
  supercategory: string;
}

interface CocoDataset {
  images: CocoImage[];
  annotations: CocoAnnotation[];
  categories: CocoCategory[];
}

// -------- Functions --------
function loadGeoJsonSplits(inputDir: string): { [split: string]: FeatureCollection } {
  const splits = ['train', 'val', 'test'];
  const datasets: { [split: string]: FeatureCollection } = {};

  for (const split of splits) {
    const filePath = path.join(inputDir, split, 'rooms.geo.json');
    if (fs.existsSync(filePath)) {
      const content = fs.readFileSync(filePath, 'utf8');
      datasets[split] = JSON.parse(content);
    }
  }

  return datasets;
}

function polygonToCocoSegmentation(polygon: Position[]): number[] {
  // Convert [[x,y], [x,y], ...] to [x1,y1,x2,y2,...] (flattened)
  return polygon.flat();
}

function calculatePolygonArea(polygon: Position[]): number {
  // Shoelace formula for polygon area
  let area = 0;
  for (let i = 0; i < polygon.length; i++) {
    const [x1, y1] = polygon[i];
    const [x2, y2] = polygon[(i + 1) % polygon.length];
    area += x1 * y2 - x2 * y1;
  }
  return Math.abs(area) / 2;
}

function calculateBoundingBox(polygon: Position[]): [number, number, number, number] {
  const xs = polygon.map(p => p[0]);
  const ys = polygon.map(p => p[1]);

  const x = Math.min(...xs);
  const y = Math.min(...ys);
  const width = Math.max(...xs) - x;
  const height = Math.max(...ys) - y;

  return [x, y, width, height];
}

function convertToCocoFormat(datasets: { [split: string]: FeatureCollection }): { [split: string]: CocoDataset } {
  const cocoDatasets: { [split: string]: CocoDataset } = {};

  // Common categories for all splits
  const categories: CocoCategory[] = [{
    id: 1,
    name: 'room',
    supercategory: 'indoor'
  }];

  for (const [split, geoJsonData] of Object.entries(datasets)) {
    const cocoImages: CocoImage[] = [];
    const cocoAnnotations: CocoAnnotation[] = [];

    // Create a single "image" entry (since our GeoJSON combines multiple images)
    // In a real scenario, you'd want to track which original image each room came from
    const imageId = 1;
    cocoImages.push({
      id: imageId,
      file_name: `${split}_combined.png`, // placeholder
      width: 1000, // normalized space
      height: 1000
    });

    let annotationId = 1;

    // Convert each room feature to COCO annotation
    for (const feature of geoJsonData.features) {
      if (feature.geometry.type !== 'Polygon') continue;

      const polygon = feature.geometry.coordinates[0]; // Outer ring only for now
      if (!polygon || polygon.length < 3) continue;

      // Convert normalized coordinates back to pixel space (assuming 1000x1000)
      const scaledPolygon: Position[] = polygon.map(([x, y]) => [x * 1000, y * 1000]);

      const segmentation = [polygonToCocoSegmentation(scaledPolygon)];
      const area = calculatePolygonArea(scaledPolygon);
      const bbox = calculateBoundingBox(scaledPolygon);

      cocoAnnotations.push({
        id: annotationId++,
        image_id: imageId,
        category_id: 1, // room
        segmentation,
        area,
        bbox,
        iscrowd: 0
      });
    }

    cocoDatasets[split] = {
      images: cocoImages,
      annotations: cocoAnnotations,
      categories
    };

    console.log(`${split}: ${cocoAnnotations.length} annotations from ${geoJsonData.features.length} features`);
  }

  return cocoDatasets;
}

function saveCocoDatasets(cocoDatasets: { [split: string]: CocoDataset }, outputDir: string) {
  fs.mkdirSync(outputDir, { recursive: true });

  for (const [split, dataset] of Object.entries(cocoDatasets)) {
    const splitDir = path.join(outputDir, split);
    fs.mkdirSync(splitDir, { recursive: true });

    const outputPath = path.join(splitDir, 'annotations.json');
    fs.writeFileSync(outputPath, JSON.stringify(dataset, null, 2));

    console.log(`Saved ${split} split: ${dataset.annotations.length} annotations`);
  }
}

// -------- Main --------
console.log('Converting GeoJSON to COCO format...');
console.log('Input directory:', INPUT_DIR);
console.log('Output directory:', OUTPUT_DIR);

// Load GeoJSON datasets
const geoJsonDatasets = loadGeoJsonSplits(INPUT_DIR);
console.log('\nLoaded GeoJSON datasets:');
for (const [split, data] of Object.entries(geoJsonDatasets)) {
  console.log(`- ${split}: ${data.features.length} rooms`);
}

// Convert to COCO format
const cocoDatasets = convertToCocoFormat(geoJsonDatasets);

// Save COCO datasets
fs.mkdirSync(OUTPUT_DIR, { recursive: true });
saveCocoDatasets(cocoDatasets, OUTPUT_DIR);

// Save conversion metadata
const metadata = {
  converted_at: new Date().toISOString(),
  source_format: 'GeoJSON',
  target_format: 'COCO',
  splits: Object.keys(cocoDatasets).map(split => ({
    split,
    images: cocoDatasets[split].images.length,
    annotations: cocoDatasets[split].annotations.length
  }))
};

fs.writeFileSync(path.join(OUTPUT_DIR, 'conversion_metadata.json'), JSON.stringify(metadata, null, 2));

console.log('\n✅ Conversion complete!');
console.log('COCO datasets saved to:', OUTPUT_DIR);
console.log('Metadata saved to:', path.join(OUTPUT_DIR, 'conversion_metadata.json'));

// Show example annotation
const firstSplit = Object.keys(cocoDatasets)[0];
const firstAnnotation = cocoDatasets[firstSplit].annotations[0];
console.log('\n📝 Example COCO annotation:');
console.log(JSON.stringify(firstAnnotation, null, 2));
