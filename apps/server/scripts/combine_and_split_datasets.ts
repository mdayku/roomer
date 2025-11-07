/*
  Combine COCO and SVG GeoJSON datasets and create train/val/test splits

  Usage:
    pnpm --filter @app/server tsx scripts/combine_and_split_datasets.ts \
      --coco-dir cubicasa5k_geojson \
      --svg-dir cubicasa5k_svg_geojson \
      --output-dir room_detection_dataset \
      --train-ratio 0.7 \
      --val-ratio 0.2 \
      --test-ratio 0.1
*/

import fs from 'node:fs';
import path from 'node:path';
import yargs from 'yargs';
import { hideBin } from 'yargs/helpers';
import type { FeatureCollection, Feature } from '../src/types/geo.js';

const argv = yargs(hideBin(process.argv))
  .option('coco-dir', { type:'string', demandOption:true, desc:'Directory with COCO GeoJSON files' })
  .option('svg-dir', { type:'string', demandOption:false, desc:'Directory with SVG GeoJSON files' })
  .option('output-dir', { type:'string', demandOption:true, desc:'Output directory for splits' })
  .option('train-ratio', { type:'number', default:0.7, desc:'Training set ratio' })
  .option('val-ratio', { type:'number', default:0.2, desc:'Validation set ratio' })
  .option('test-ratio', { type:'number', default:0.1, desc:'Test set ratio' })
  .parseSync();

const COCO_DIR = path.resolve(argv['coco-dir']);
const SVG_DIR = argv['svg-dir'] ? path.resolve(argv['svg-dir']) : null;
const OUTPUT_DIR = path.resolve(argv['output-dir']);
const TRAIN_RATIO = Number(argv['train-ratio']);
const VAL_RATIO = Number(argv['val-ratio']);
const TEST_RATIO = Number(argv['test-ratio']);

if (Math.abs(TRAIN_RATIO + VAL_RATIO + TEST_RATIO - 1) > 0.001) {
  console.error('Error: Train/Val/Test ratios must sum to 1.0');
  process.exit(1);
}

fs.mkdirSync(OUTPUT_DIR, { recursive: true });
fs.mkdirSync(path.join(OUTPUT_DIR, 'train'), { recursive: true });
fs.mkdirSync(path.join(OUTPUT_DIR, 'val'), { recursive: true });
fs.mkdirSync(path.join(OUTPUT_DIR, 'test'), { recursive: true });

// -------- Functions --------
function loadGeoJsonFiles(dir: string): FeatureCollection[] {
  const files = fs.readdirSync(dir).filter(f => f.endsWith('.geo.json'));
  console.log(`Loading ${files.length} files from ${dir}`);

  const collections: FeatureCollection[] = [];
  for (const file of files) {
    try {
      const content = fs.readFileSync(path.join(dir, file), 'utf8');
      const fc: FeatureCollection = JSON.parse(content);
      collections.push(fc);
    } catch (err) {
      console.warn(`Warning: Failed to load ${file}:`, err);
    }
  }
  return collections;
}

function shuffleArray<T>(array: T[]): T[] {
  const shuffled = [...array];
  for (let i = shuffled.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]];
  }
  return shuffled;
}

function splitCollections(collections: FeatureCollection[], trainRatio: number, valRatio: number) {
  const shuffled = shuffleArray(collections);

  const trainCount = Math.floor(collections.length * trainRatio);
  const valCount = Math.floor(collections.length * valRatio);

  const train = shuffled.slice(0, trainCount);
  const val = shuffled.slice(trainCount, trainCount + valCount);
  const test = shuffled.slice(trainCount + valCount);

  return { train, val, test };
}

function saveCollections(collections: FeatureCollection[], outputPath: string) {
  // Combine all collections into one big dataset
  const allFeatures: Feature[] = [];
  for (const fc of collections) {
    // Add source metadata to each feature
    for (const feature of fc.features) {
      feature.properties.source = fc.image?.file_name || 'unknown';
      feature.properties.image_width = fc.image?.width || 1000;
      feature.properties.image_height = fc.image?.height || 1000;
      allFeatures.push(feature);
    }
  }

  const combined: FeatureCollection = {
    type: 'FeatureCollection',
    image: {
      normalized: true,
      total_features: allFeatures.length
    },
    features: allFeatures
  };

  fs.writeFileSync(outputPath, JSON.stringify(combined, null, 2));
  return allFeatures.length;
}

// -------- Main --------
console.log('Combining COCO and SVG datasets...');

// Load COCO data
const cocoCollections = loadGeoJsonFiles(COCO_DIR);

// Load SVG data if available
let svgCollections: FeatureCollection[] = [];
if (SVG_DIR && fs.existsSync(SVG_DIR)) {
  svgCollections = loadGeoJsonFiles(SVG_DIR);
}

// Combine all collections
const allCollections = [...cocoCollections, ...svgCollections];
console.log(`\nTotal collections: ${allCollections.length}`);
console.log(`- COCO: ${cocoCollections.length} collections`);
console.log(`- SVG: ${svgCollections.length} collections`);

// Calculate total rooms
let totalRooms = 0;
for (const fc of allCollections) {
  totalRooms += fc.features.length;
}
console.log(`Total rooms: ${totalRooms}`);

// Split into train/val/test
const { train, val, test } = splitCollections(allCollections, TRAIN_RATIO, VAL_RATIO);

console.log('\nSplitting into train/val/test sets...');
console.log(`Train: ${train.length} collections (${(TRAIN_RATIO * 100).toFixed(1)}%)`);
console.log(`Val: ${val.length} collections (${(VAL_RATIO * 100).toFixed(1)}%)`);
console.log(`Test: ${test.length} collections (${(TEST_RATIO * 100).toFixed(1)}%)`);

// Save splits
const trainRooms = saveCollections(train, path.join(OUTPUT_DIR, 'train', 'rooms.geo.json'));
const valRooms = saveCollections(val, path.join(OUTPUT_DIR, 'val', 'rooms.geo.json'));
const testRooms = saveCollections(test, path.join(OUTPUT_DIR, 'test', 'rooms.geo.json'));

console.log('\n✅ Dataset splits saved:');
console.log(`- Train: ${trainRooms} rooms → ${path.join(OUTPUT_DIR, 'train', 'rooms.geo.json')}`);
console.log(`- Val: ${valRooms} rooms → ${path.join(OUTPUT_DIR, 'val', 'rooms.geo.json')}`);
console.log(`- Test: ${testRooms} rooms → ${path.join(OUTPUT_DIR, 'test', 'rooms.geo.json')}`);

// Save metadata
const metadata = {
  created_at: new Date().toISOString(),
  total_collections: allCollections.length,
  total_rooms: totalRooms,
  splits: {
    train: {
      collections: train.length,
      rooms: trainRooms,
      ratio: TRAIN_RATIO
    },
    val: {
      collections: val.length,
      rooms: valRooms,
      ratio: VAL_RATIO
    },
    test: {
      collections: test.length,
      rooms: testRooms,
      ratio: TEST_RATIO
    }
  },
  sources: {
    coco: cocoCollections.length,
    svg: svgCollections.length
  }
};

fs.writeFileSync(
  path.join(OUTPUT_DIR, 'metadata.json'),
  JSON.stringify(metadata, null, 2)
);

console.log(`\n📊 Metadata saved to: ${path.join(OUTPUT_DIR, 'metadata.json')}`);
console.log('\n🎉 Dataset preparation complete! Ready for training.');
