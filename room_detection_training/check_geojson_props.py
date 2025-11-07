#!/usr/bin/env python3
"""
Check GeoJSON feature properties for image association
"""

import json

def main():
    with open('../room_detection_dataset_clean/train/rooms.geo.json', 'r') as f:
        data = json.load(f)

    print('GeoJSON properties:', data.get('image', {}))

    # Check if any features have image information
    for i, feature in enumerate(data['features'][:5]):
        props = feature['properties']
        print(f'Feature {i} properties keys: {list(props.keys())}')
        if 'image_width' in props:
            print(f'  Image dimensions: {props["image_width"]}x{props["image_height"]}')
        if 'source' in props:
            print(f'  Source: {props["source"]}')

    # Check for unique image dimensions to see if we can group by image
    dimensions = set()
    for feature in data['features'][:100]:  # Check first 100
        props = feature['properties']
        if 'image_width' in props and 'image_height' in props:
            dimensions.add((props['image_width'], props['image_height']))

    print(f'Unique image dimensions found: {dimensions}')

if __name__ == "__main__":
    main()
