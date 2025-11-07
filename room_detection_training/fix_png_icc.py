#!/usr/bin/env python3
"""
Fix ICC color profile issues in PNG images
Removes problematic ICC profiles that cause warnings
"""

import os
import sys
from pathlib import Path
from PIL import Image
import warnings

def fix_png_icc(input_path, output_path=None, overwrite=False):
    """Fix ICC profile issues in a single PNG file"""
    if output_path is None:
        if overwrite:
            output_path = input_path
        else:
            # Create a new filename with _fixed suffix
            stem = input_path.stem
            suffix = input_path.suffix
            output_path = input_path.parent / f"{stem}_fixed{suffix}"

    try:
        # Open image with PIL
        with Image.open(input_path) as img:
            # Remove ICC profile to avoid warnings
            if 'icc_profile' in img.info:
                img_without_icc = Image.new(img.mode, img.size)
                img_without_icc.putdata(img.getdata())

                # Copy basic info but remove ICC profile
                info_copy = img.info.copy()
                if 'icc_profile' in info_copy:
                    del info_copy['icc_profile']

                img_without_icc.save(output_path, **info_copy)
                print(f"Fixed: {input_path.name} -> {output_path.name}")
                return True
            else:
                # No ICC profile to fix, just copy if needed
                if not overwrite:
                    img.save(output_path)
                    print(f"No ICC profile found: {input_path.name}")
                return True

    except Exception as e:
        print(f"Error processing {input_path.name}: {e}")
        return False

def fix_directory_pngs(directory_path, recursive=True, overwrite=False):
    """Fix all PNG files in a directory"""
    directory = Path(directory_path)

    if not directory.exists():
        print(f"Directory not found: {directory}")
        return

    # Find all PNG files
    if recursive:
        png_files = list(directory.rglob("*.png"))
    else:
        png_files = list(directory.glob("*.png"))

    if not png_files:
        print(f"No PNG files found in {directory}")
        return

    print(f"Found {len(png_files)} PNG files to process...")

    fixed_count = 0
    for png_file in png_files:
        if fix_png_icc(png_file, overwrite=overwrite):
            fixed_count += 1

    print(f"Processed {fixed_count}/{len(png_files)} PNG files successfully")

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Fix ICC color profile issues in PNG images")
    parser.add_argument("path", help="Path to PNG file or directory containing PNG files")
    parser.add_argument("--recursive", "-r", action="store_true",
                       help="Recursively process subdirectories")
    parser.add_argument("--overwrite", "-o", action="store_true",
                       help="Overwrite original files instead of creating _fixed versions")

    args = parser.parse_args()

    path = Path(args.path)

    if path.is_file():
        if path.suffix.lower() == '.png':
            fix_png_icc(path, overwrite=args.overwrite)
        else:
            print("File must be a PNG file")
            sys.exit(1)
    elif path.is_dir():
        fix_directory_pngs(path, recursive=args.recursive, overwrite=args.overwrite)
    else:
        print(f"Path not found: {path}")
        sys.exit(1)

if __name__ == "__main__":
    # Suppress PIL warnings
    warnings.filterwarnings("ignore", category=UserWarning, module="PIL")

    main()
