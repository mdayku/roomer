#!/usr/bin/env python3
"""
Check the content of label files
"""

import os
from pathlib import Path

def main():
    labels_dir = Path("yolo_data/train/labels")
    if not labels_dir.exists():
        print(f"Labels directory doesn't exist: {labels_dir.absolute()}")
        return

    label_files = list(labels_dir.glob("*.txt"))
    print(f"Found {len(label_files)} label files")

    if not label_files:
        print("No label files found!")
        return

    print("\nFirst 5 label files:")
    for i, label_file in enumerate(label_files[:5]):
        print(f"\n{i+1}. {label_file.name}:")
        try:
            with open(label_file, 'r') as f:
                content = f.read().strip()
                print(f"   Content: '{content[:200]}'")
                lines = content.split('\n')
                print(f"   Lines: {len(lines)}")
                if lines and lines[0]:
                    parts = lines[0].split()
                    print(f"   First line parts: {len(parts)}")
                    if len(parts) > 1:
                        print(f"   Class ID: {parts[0]}")
                        print(f"   Coordinates: {len(parts)-1}")
        except Exception as e:
            print(f"   Error reading file: {e}")

if __name__ == "__main__":
    main()
