import os
import glob
import json
import cv2
import numpy as np
import yaml
import argparse
from pathlib import Path

# Set environment variable to allow unsafe loading (for trusted models)
os.environ['TORCH_LOAD_WEIGHTS_ONLY'] = 'False'

def coco_to_yolo_bbox(coco_bbox, img_width, img_height):
    """
    Convert COCO bbox format [x, y, width, height] to YOLO format [x_center, y_center, width, height] normalized
    """
    x, y, w, h = coco_bbox
    x_center = (x + w / 2) / img_width
    y_center = (y + h / 2) / img_height
    width = w / img_width
    height = h / img_height
    return [x_center, y_center, width, height]

def load_coco_annotations(coco_file):
    """
    Load COCO format annotations
    """
    with open(coco_file, 'r') as f:
        coco_data = json.load(f)
    
    # Create mapping from category_id to category name
    categories = {cat['id']: cat['name'] for cat in coco_data['categories']}
    
    # Create mapping from image_id to image info
    images = {img['id']: img for img in coco_data['images']}
    
    # Group annotations by image_id
    image_annotations = {}
    for ann in coco_data['annotations']:
        img_id = ann['image_id']
        if img_id not in image_annotations:
            image_annotations[img_id] = []
        image_annotations[img_id].append(ann)
    
    return categories, images, image_annotations

def get_class_id_mapping():
    """
    Define mapping from COCO class names to YOLO class IDs for face and license plate detection
    """
    # You may need to adjust these mappings based on your COCO annotation class names
    class_mapping = {
        'person': 0,  # faces will be detected from person class
        'face': 0,    # if you have specific face annotations
        'license_plate': 1,  # if you have license plate annotations
        'car': -1,    # ignore cars (we only care about faces and license plates)
        'truck': -1,  # ignore trucks
        'bus': -1     # ignore buses
    }
    return class_mapping

def process_images_with_coco_labels(config, coco_file):
    """
    Process images using existing COCO labels instead of running YOLO detection
    """
    print(f"Loading COCO annotations from: {coco_file}")
    categories, images, image_annotations = load_coco_annotations(coco_file)
    class_mapping = get_class_id_mapping()
    
    # Create output directories
    os.makedirs("annot_txt", exist_ok=True)
    os.makedirs(config['output_folder'], exist_ok=True)
    
    # Get all images
    image_files = sorted(glob.glob(config['images_path'] + "/*" + config["img_format"]))
    
    processed_count = 0
    for img_path in image_files:
        img_name = os.path.basename(img_path)
        img_name_no_ext = os.path.splitext(img_name)[0]
        
        # Find corresponding image in COCO data
        matching_image = None
        matching_img_id = None
        for img_id, img_info in images.items():
            if img_info['file_name'] == img_name:
                matching_image = img_info
                matching_img_id = img_id
                break
        
        if matching_image is None:
            print(f"Warning: No COCO annotation found for {img_name}, skipping...")
            continue
        
        # Get annotations for this image
        annotations = image_annotations.get(matching_img_id, [])
        
        # Convert COCO annotations to YOLO format
        yolo_annotations = []
        for ann in annotations:
            cat_name = categories[ann['category_id']]
            if cat_name in class_mapping and class_mapping[cat_name] >= 0:
                class_id = class_mapping[cat_name]
                coco_bbox = ann['bbox']
                yolo_bbox = coco_to_yolo_bbox(coco_bbox, matching_image['width'], matching_image['height'])
                yolo_annotations.append([class_id] + yolo_bbox)
        
        # Save YOLO format annotations
        yolo_file = f"annot_txt/{img_name_no_ext}.txt"
        with open(yolo_file, 'w') as f:
            for ann in yolo_annotations:
                f.write(f"{ann[0]} {ann[1]:.6f} {ann[2]:.6f} {ann[3]:.6f} {ann[4]:.6f}\n")
        
        processed_count += 1
        if processed_count % 50 == 0:
            print(f"Processed {processed_count}/{len(image_files)} images...")
    
    print(f"Converted {processed_count} COCO annotations to YOLO format")
    return processed_count

def apply_blur_from_annotations(config):
    """
    Apply blur to images based on existing annotations
    """
    image_folder = config['images_path']
    output_folder = config['output_folder']
    
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Get all annotation files
    annotation_files = glob.glob("annot_txt/*.txt")
    
    for txt_file in annotation_files:
        txt_filename = os.path.basename(txt_file)
        image_file = txt_filename.replace('.txt', config["img_format"])
        image_path = os.path.join(image_folder, image_file)
        
        if not os.path.exists(image_path):
            print(f"Warning: Image file {image_path} not found")
            continue
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Warning: Could not load image {image_path}")
            continue
        
        img_height, img_width = image.shape[:2]
        
        # Read annotations
        with open(txt_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split()
                if len(parts) != 5:
                    continue
                
                class_id, x_center, y_center, width, height = map(float, parts)
                
                # Convert YOLO format back to pixel coordinates
                x1 = int((x_center - width/2) * img_width)
                y1 = int((y_center - height/2) * img_height)
                x2 = int((x_center + width/2) * img_width)
                y2 = int((y_center + height/2) * img_height)
                
                # Ensure coordinates are within image bounds
                x1 = max(0, min(x1, img_width-1))
                y1 = max(0, min(y1, img_height-1))
                x2 = max(0, min(x2, img_width-1))
                y2 = max(0, min(y2, img_height-1))
                
                # Apply blur to the region
                if x2 > x1 and y2 > y1:
                    roi = image[y1:y2, x1:x2]
                    blurred_roi = cv2.GaussianBlur(roi, (config['blur_radius'], config['blur_radius']), 0)
                    image[y1:y2, x1:x2] = blurred_roi
        
        # Save blurred image
        output_path = os.path.join(output_folder, f"{os.path.splitext(image_file)[0]}_blurred.jpg")
        cv2.imwrite(output_path, image)
    
    print(f"@@ The blurred images are saved in Directory -------> {config['output_folder']}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="path of the configuration file", required=True)
    parser.add_argument("--coco", help="path of the COCO annotation file", required=True)
    parser.add_argument("--skip-detection", action="store_true", 
                       help="Skip YOLO detection and use only COCO annotations")
    args = parser.parse_args()
    
    # Clean up previous runs
    if os.path.exists("annot_txt"):
        import shutil
        shutil.rmtree("annot_txt")
    
    # Read configuration file
    with open(args.config, 'r') as f:
        try:
            config = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            print(exc)
            return
    
    if args.skip_detection:
        print("Using only COCO annotations, skipping YOLO detection...")
        process_images_with_coco_labels(config, args.coco)
    else:
        print("Running YOLO detection first, then merging with COCO annotations...")
        # First run YOLO detection (existing code)
        from ultralytics import YOLO
        model = YOLO(config["model_path"])
        
        if config["gpu_avail"]:
            _ = model(source=config['images_path'],
                    save=False,
                    save_txt=True,
                    conf=config['detection_conf_thresh'],
                    device='cuda:0',
                    project='runs/detect/',
                    name="yolo_images_pred")
        else:
            _ = model(source=config['images_path'],
                    save=False,
                    save_txt=True,
                    conf=config['detection_conf_thresh'],
                    device='cpu',
                    project="runs/detect/",
                    name="yolo_images_pred")
        
        # Then merge with COCO annotations
        # TODO: Implement merging logic here
        
    # Apply blur based on annotations
    apply_blur_from_annotations(config)

if __name__ == "__main__":
    main()
