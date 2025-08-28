import os
import glob
import json
import cv2
import pybboxes as pbx
import yaml
import argparse
from ultralytics import YOLO


parser = argparse.ArgumentParser()
parser.add_argument("--config", help = "path of the training configuartion file", required = True)
args = parser.parse_args()

if (os.path.exists("annot_txt")):
    import shutil
    shutil.rmtree("annot_txt")

#Reading the configuration file
with open(args.config, 'r') as f:
    try:
        config = yaml.safe_load(f)
    except yaml.YAMLError as exc:
        print(exc)

print(f"Loading model from: {config['model_path']}")
model = YOLO(config["model_path"])

# Support multiple image formats
supported_formats = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
all_images = []
for fmt in supported_formats:
    all_images.extend(glob.glob(config['images_path'] + "/*" + fmt))
    all_images.extend(glob.glob(config['images_path'] + "/*" + fmt.upper()))

print(f"Found {len(all_images)} images in {config['images_path']}")

if len(all_images) == 0:
    print("No images found! Exiting...")
    exit(1)

if(config["gpu_avail"]):
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


#images = [int(item.split("/")[1].replace(config['img_format'], "")) for item in images]
# Use the previously found image list
images = sorted(all_images)

os.mkdir("annot_txt")

annot_dir = f'runs/detect/yolo_images_pred/labels/'

try:
    for file in os.listdir(annot_dir):
        if (file.endswith('.txt')):
            #frame_num = int(file.replace(".txt","").split("_")[1])
            with open(annot_dir+file, 'r') as fin:
                for line in fin.readlines():
                    line = [float(item) for item in line.split()[1:]]
                    line = pbx.convert_bbox(line, from_type="yolo", to_type="voc", image_size=(config["img_width"], config["img_height"]))
                    data_string = " ".join(str(num) for num in line)
                    with open(f"annot_txt/{os.path.basename(file)}", "a") as f:
                        f.write(data_string+"\n")
except Exception as e:
    print(f'Error processing annotations: {e}')


def blur_regions(image, regions):
    """
    Blurs the image, given the x1,y1,x2,y2 cordinates using Gaussian Blur.
    """
    for region in regions:
        x1,y1,x2,y2 = region
        x1, y1, x2, y2 = round(x1), round(y1), round(x2), round(y2)
        roi = image[y1:y2, x1:x2]
        blurred_roi = cv2.GaussianBlur(roi, (config['blur_radius'], config['blur_radius']), 0)
        image[y1:y2, x1:x2] = blurred_roi
    return image

txt_folder = 'annot_txt/'
image_folder = config['images_path']
output_folder = config['output_folder']

# Create the output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# List all text files in the 'dir' folder
txt_files = [f for f in os.listdir(txt_folder) if f.endswith('.txt')]

print(f"Processing {len(txt_files)} annotation files...")

for txt_file in txt_files:
    # Read the text file containing bounding box information
    with open(os.path.join(txt_folder, txt_file), 'r') as f:
        lines = f.readlines()

    # Extract bounding box coordinates from the txt file
    bboxes = []
    for line in lines:
        values = line.strip().split()
        if len(values) >= 4:  # Ensure sufficient coordinate values
            x_min, y_min, x_max, y_max = map(int, values[:4])  # Assuming VOC format with x_min, y_min, x_max, y_max
            bboxes.append([x_min, y_min, x_max, y_max])

    # Find corresponding image file - support multiple formats
    base_name = os.path.splitext(txt_file)[0]
    image_file = None
    for fmt in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.JPG', '.JPEG', '.PNG', '.BMP', '.TIFF']:
        potential_image = base_name + fmt
        image_path = os.path.join(image_folder, potential_image)
        if os.path.exists(image_path):
            image_file = potential_image
            break
    
    if image_file is None:
        print(f"No corresponding image found for {txt_file}")
        continue

    # Read the corresponding image
    image_path = os.path.join(image_folder, image_file)
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"Failed to load image: {image_path}")
        continue

    # Apply Gaussian blur to each bounding box region
    if bboxes:
        image = blur_regions(image, bboxes)

    # Save the blurred image to the output folder
    output_file = base_name + '_blurred.jpg'
    output_path = os.path.join(output_folder, output_file)
    
    success = cv2.imwrite(output_path, image)
    if success:
        print(f"Processed: {image_file} -> {output_file}")
    else:
        print(f"Failed to save: {output_file}")

print(f"@@ The blurred images are saved in Directory -------> {config['output_folder']}")