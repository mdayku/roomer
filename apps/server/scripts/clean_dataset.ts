/*
  Clean dataset by removing collections with no room features

  Usage:
    pnpm --filter @app/server tsx scripts/clean_dataset.ts \
      --input-dir room_detection_dataset \
      --output-dir room_detection_dataset_clean
*/

import fs from 'node:fs';
import path from 'node:path';
import yargs from 'yargs';
import { hideBin } from 'yargs/helpers';
import type { FeatureCollection } from '../src/types/geo.js';

const argv = yargs(hideBin(process.argv))
  .option('input-dir', { type:'string', demandOption:true, desc:'Input dataset directory' })
  .option('output-dir', { type:'string', demandOption:true, desc:'Output cleaned dataset directory' })
  .parseSync();

const INPUT_DIR = path.resolve(argv['input-dir']);
const OUTPUT_DIR = path.resolve(argv['output-dir']);

// -------- Functions --------
function loadCollections(dir: string): FeatureCollection[] {
  const collections: FeatureCollection[] = [];
  const splits = ['train', 'val', 'test'];

  for (const split of splits) {
    const splitDir = path.join(dir, split);
    const filePath = path.join(splitDir, 'rooms.geo.json');

    if (fs.existsSync(filePath)) {
      const content = fs.readFileSync(filePath, 'utf8');
      const collection: FeatureCollection = JSON.parse(content);
      collections.push({ ...collection, _split: split });
    }
  }

  return collections;
}

function filterValidCollections(collections: FeatureCollection[]): FeatureCollection[] {
  return collections.filter(collection => {
    const hasRooms = collection.features && collection.features.length > 0;
    if (!hasRooms) {
      console.log(`Removing empty collection from ${collection._split} split`);
    }
    return hasRooms;
  });
}

function saveCollectionsBySplit(collections: FeatureCollection[], outputDir: string) {
  const splits = { train: [], val: [], test: [] };

  // Group by split
  for (const collection of collections) {
    const split = collection._split;
    delete collection._split; // Remove temporary property
    splits[split].push(collection);
  }

  // Save each split
  for (const [split, splitCollections] of Object.entries(splits)) {
    if (splitCollections.length === 0) continue;

    const splitDir = path.join(outputDir, split);
    fs.mkdirSync(splitDir, { recursive: true });

    // Combine all collections in this split into one file
    const allFeatures = splitCollections.flatMap(c => c.features);
    const combined: FeatureCollection = {
      type: 'FeatureCollection',
      image: {
        normalized: true,
        total_features: allFeatures.length
      },
      features: allFeatures
    };

    const outputPath = path.join(splitDir, 'rooms.geo.json');
    fs.writeFileSync(outputPath, JSON.stringify(combined, null, 2));

    console.log(`Saved ${split} split: ${allFeatures.length} rooms from ${splitCollections.length} collections`);
  }

  return splits;
}

// -------- Main --------
console.log('Loading dataset from:', INPUT_DIR);

const collections = loadCollections(INPUT_DIR);
console.log(`\nLoaded ${collections.length} collections total`);

const validCollections = filterValidCollections(collections);
console.log(`\nAfter filtering: ${validCollections.length} valid collections`);

if (validCollections.length === 0) {
  console.error('No valid collections found!');
  process.exit(1);
}

// Group by split and save
fs.mkdirSync(OUTPUT_DIR, { recursive: true });
const splits = saveCollectionsBySplit(validCollections, OUTPUT_DIR);

// Calculate statistics
let totalRooms = 0;
let totalCollections = 0;

for (const [split, collections] of Object.entries(splits)) {
  if (collections.length > 0) {
    const roomCount = collections.flatMap(c => c.features).length;
    totalRooms += roomCount;
    totalCollections += collections.length;
    console.log(`${split}: ${collections.length} collections, ${roomCount} rooms`);
  }
}

console.log(`\n✅ Dataset cleaned successfully!`);
console.log(`Total collections: ${totalCollections}`);
console.log(`Total rooms: ${totalRooms}`);
console.log(`Output directory: ${OUTPUT_DIR}`);

// Save updated metadata
const metadataPath = path.join(INPUT_DIR, 'metadata.json');
if (fs.existsSync(metadataPath)) {
  const metadata = JSON.parse(fs.readFileSync(metadataPath, 'utf8'));
  metadata.cleaned_at = new Date().toISOString();
  metadata.after_cleaning = {
    total_collections: totalCollections,
    total_rooms: totalRooms
  };

  fs.writeFileSync(path.join(OUTPUT_DIR, 'metadata.json'), JSON.stringify(metadata, null, 2));
}
