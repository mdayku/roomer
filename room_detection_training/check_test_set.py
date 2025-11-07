#!/usr/bin/env python3
"""
Check what's actually in the test set
"""

from pathlib import Path

def main():
    test_labels_dir = Path('./yolo_data/test/labels')
    test_images_dir = Path('./yolo_data/test/images')

    print('Checking test set composition...')
    print(f'Label files: {len(list(test_labels_dir.glob("*.txt")))}')
    print(f'Image files: {len(list(test_images_dir.glob("*.png")))}')

    # Check how many label files have actual content
    non_empty_labels = 0
    total_gt_rooms = 0
    gt_counts = []

    for label_file in test_labels_dir.glob('*.txt'):
        try:
            with open(label_file, 'r') as f:
                lines = [line.strip() for line in f.readlines() if line.strip()]
                gt_count = len(lines)
                if gt_count > 0:
                    non_empty_labels += 1
                    total_gt_rooms += gt_count
                    gt_counts.append(gt_count)
        except:
            continue

    print(f'\nLabels with rooms: {non_empty_labels}')
    print(f'Total GT rooms counted: {total_gt_rooms}')
    if gt_counts:
        print(f'Average rooms per image: {total_gt_rooms/non_empty_labels:.1f}')
        print(f'Max rooms in image: {max(gt_counts)}')
        print(f'Min rooms in image: {min(gt_counts)}')

    # Check if there are images without labels
    images_without_labels = []
    for img_file in test_images_dir.glob('*.png'):
        label_file = test_labels_dir / f'{img_file.stem}.txt'
        if not label_file.exists():
            images_without_labels.append(img_file.name)

    print(f'\nImages without labels: {len(images_without_labels)}')
    if images_without_labels:
        print('Sample:', images_without_labels[:3])

    # Check some specific files
    print('\nChecking specific files:')
    test_files = ['000322_F1_original.txt', '000343_F1_original.txt', '000001_F1_original.txt']
    for filename in test_files:
        filepath = test_labels_dir / filename
        if filepath.exists():
            with open(filepath, 'r') as f:
                lines = [line.strip() for line in f.readlines() if line.strip()]
                print(f'{filename}: {len(lines)} rooms')
                if lines:
                    print(f'  Sample line: {lines[0]}')
        else:
            print(f'{filename}: FILE NOT FOUND')

if __name__ == "__main__":
    main()
