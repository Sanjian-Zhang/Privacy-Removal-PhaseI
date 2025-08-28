#!/usr/bin/env python3
"""
Script for validating image and coordinate file formats
"""

import os
import argparse
import glob
from pathlib import Path

def validate_txt_format(txt_path):
    """Validate txt file format"""
    errors = []
    
    try:
        with open(txt_path, 'r') as f:
            lines = f.readlines()
        
        for i, line in enumerate(lines, 1):
            line = line.strip()
            if not line:
                continue
                
            coords = line.split(',')
            if len(coords) != 8:
                errors.append(f"Line {i}: Should have 8 coordinates, actual {len(coords)}")
                continue
            
            try:
                coords = [int(x) for x in coords]
            except ValueError:
                errors.append(f"Line {i}: Coordinates must be integers")
                continue
            
            if any(x < 0 for x in coords):
                errors.append(f"Line {i}: Coordinates cannot be negative")
    
    except Exception as e:
        errors.append(f"Failed to read file: {e}")
    
    return errors

def validate_image_file(image_path):
    """Validate image file"""
    errors = []
    
    if not os.path.exists(image_path):
        errors.append("File does not exist")
        return errors
    
    size = os.path.getsize(image_path)
    if size == 0:
        errors.append("File is empty")
    elif size < 1024:
        errors.append("File may be corrupted (too small)")
    
    return errors

def main():
    parser = argparse.ArgumentParser(description="Validate GaRNet input data format")
    parser.add_argument("--image_dir", required=True, help="Image directory")
    parser.add_argument("--box_dir", required=True, help="Coordinate file directory")
    
    args = parser.parse_args()
    
    print("🔍 Validating GaRNet input data format")
    print("=" * 50)
    
    if not os.path.exists(args.image_dir):
        print(f"❌ Image directory does not exist: {args.image_dir}")
        return
    
    if not os.path.exists(args.box_dir):
        print(f"❌ Coordinate directory does not exist: {args.box_dir}")
        return
    
    image_files = glob.glob(os.path.join(args.image_dir, "*.jpg"))
    txt_files = glob.glob(os.path.join(args.box_dir, "*.txt"))
    
    print(f"📁 Image files: {len(image_files)}")
    print(f"📁 Coordinate files: {len(txt_files)}")
    
    image_names = {Path(f).stem for f in image_files}
    txt_names = {Path(f).stem for f in txt_files}
    
    missing_txt = image_names - txt_names
    missing_img = txt_names - image_names
    matched = image_names & txt_names
    
    print(f"\\n📊 Matching statistics:")
    print(f"   ✓ Perfect matches: {len(matched)} pairs")
    if missing_txt:
        print(f"   ⚠️  Missing coordinate files: {len(missing_txt)}")
        for name in sorted(missing_txt):
            print(f"      - {name}.txt")
    
    if missing_img:
        print(f"   ⚠️  Missing image files: {len(missing_img)}")
        for name in sorted(missing_img):
            print(f"      - {name}.jpg")
    
    print(f"\\n🔍 Detailed validation:")
    total_errors = 0
    
    for name in sorted(matched):
        image_path = os.path.join(args.image_dir, f"{name}.jpg")
        txt_path = os.path.join(args.box_dir, f"{name}.txt")
        
        print(f"\\n📄 {name}:")
        
        img_errors = validate_image_file(image_path)
        if img_errors:
            print(f"   ❌ Image issues:")
            for error in img_errors:
                print(f"      - {error}")
            total_errors += len(img_errors)
        else:
            print(f"   ✓ Image format correct")
        
        txt_errors = validate_txt_format(txt_path)
        if txt_errors:
            print(f"   ❌ Coordinate file issues:")
            for error in txt_errors:
                print(f"      - {error}")
            total_errors += len(txt_errors)
        else:
            with open(txt_path, 'r') as f:
                lines = [l.strip() for l in f.readlines() if l.strip()]
            print(f"   ✓ Coordinate format correct ({len(lines)} text regions)")
    
    print(f"\\n📋 Validation summary:")
    print(f"   Total file pairs: {len(matched)}")
    print(f"   Error count: {total_errors}")
    
    if total_errors == 0 and len(matched) > 0:
        print(f"   🎉 All file formats are correct, ready to process!")
        print(f"\\n💡 Run command:")
        print(f"   python process_my_images.py --input_dir {args.image_dir} --box_dir {args.box_dir} --output_dir ./results")
    elif total_errors > 0:
        print(f"   ⚠️  Please fix the above errors and validate again")
    else:
        print(f"   ❌ No matching file pairs found")

if __name__ == "__main__":
    main()
