#!/usr/bin/env python3
"""
Check GeoJSON data structure
"""

import json

def main():
    with open('../room_detection_dataset_clean/train/rooms.geo.json', 'r') as f:
        data = json.load(f)

    print('Total features:', len(data['features']))

    # Check first few features
    for i, feature in enumerate(data['features'][:2]):
        print(f'Feature {i}:')
        print(f'  ID: {feature["properties"]["id"]}')
        print(f'  Geometry type: {feature["geometry"]["type"]}')
        if feature['geometry']['type'] == 'Polygon':
            coords = feature['geometry']['coordinates'][0]  # Outer ring
            print(f'  Polygon points: {len(coords)}')
            print(f'  First 3 points: {coords[:3]}')
            print(f'  Normalized: {feature["properties"].get("normalized", "unknown")}')
        print()

if __name__ == "__main__":
    main()
