import os
import json
import cv2
import shutil
import random
from collections import defaultdict
import matplotlib.pyplot as plt

# Define paths for the dataset
TACO_DATA_PATH = r"C:\Users\User\Desktop\Noa Project\Taco\TACO-master\data"
ANNOTATIONS_FILE = os.path.join(TACO_DATA_PATH, "annotations.json")
OUTPUT_DATASET_DIR = os.path.join(TACO_DATA_PATH, "TacoCropped")

# Categories for TrashNet
TRASHNET_CATEGORIES = ["plastic", "metal", "paper", "glass", "cardboard", "other"]  # Replaced "trash" with "other"

# Create directories for each category if they don't exist
for cat in TRASHNET_CATEGORIES:
    os.makedirs(os.path.join(OUTPUT_DATASET_DIR, cat), exist_ok=True)

# Dictionary to count the number of images per category
category_counts = defaultdict(int)

# Function to map category names
def map_category_to_trashnet(name: str, supercat: str):
    name_lower = name.strip().lower()
    super_lower = supercat.strip().lower()
    
    # Dictionary for mapping
    NAME_TO_CATEGORY = {
        "aluminium blister pack": "plastic",
        "carded blister pack": "plastic",
        "other plastic bottle": "plastic",  # Correctly mapped to "plastic"
        "clear plastic bottle": "plastic",
        "plastic bottle cap": "plastic",
        "disposable plastic cup": "plastic",
        "foam cup": "plastic",
        "other plastic cup": "plastic",
        "plastic lid": "plastic",
        "plastic film": "plastic",
        "garbage bag": "plastic",
        "other plastic wrapper": "plastic",
        "single-use carrier bag": "plastic",
        "plastic container": "plastic",
        "foam food container": "plastic",
        "other plastic container": "plastic",
        "plastic utensils": "plastic",
        "plastic straw": "plastic",
        "styrofoam piece": "plastic",
        "crisp packet": "plastic",
        "six pack rings": "plastic",
        "spread tub": "plastic",
        "tupperware": "plastic",
        "metal bottle cap": "metal",
        "scrap metal": "metal",
        "pop tab": "metal",
        "drink can": "metal",
        "food can": "metal",
        "aerosol": "metal",
        "paper cup": "paper",
        "normal paper": "paper",
        "magazine paper": "paper",
        "tissues": "paper",
        "wrapping paper": "paper",
        "paper bag": "paper",
        "paper straw": "paper",
        "other carton": "cardboard",
        "egg carton": "cardboard",
        "drink carton": "cardboard",
        "corrugated carton": "cardboard",
        "meal carton": "cardboard",
        "pizza box": "cardboard",
        "toilet tube": "cardboard",
        "glass bottle": "glass",
        "broken glass": "glass",
        "glass jar": "glass",
        "glass cup": "glass",
        "cigarette": "other",  # Replaced "trash" with "other"
        "food waste": "other",  # Replaced "trash" with "other"
        "battery": "other",  # Replaced "trash" with "other"
        "shoe": "other",  # Replaced "trash" with "other"
    }
    if name_lower in NAME_TO_CATEGORY:
        return NAME_TO_CATEGORY[name_lower]
    
    # Mapping based on supercategory
    SUPER_TO_CATEGORY = {
        "bottle": "plastic",
        "bottle cap": "plastic",
        "cup": "plastic",
        "carton": "cardboard",
        "can": "metal",
        "paper": "paper",
        "paper bag": "paper",
        "plastic bag & wrapper": "plastic",
        "plastic container": "plastic",
        "other plastic": "plastic",  # Correctly mapped to "plastic"
        # Add other mappings as needed...
    }
    for key, value in SUPER_TO_CATEGORY.items():
        if key in super_lower:
            return value
    return "other"  # Default to "other" if no match

# Function to resize images while maintaining aspect ratio
def resize_keep_aspect(img, desired_size=224):
    old_size = img.shape[:2]  # (height, width)
    ratio = float(desired_size) / max(old_size)
    new_size = tuple([int(x * ratio) for x in old_size])
    resized = cv2.resize(img, (new_size[1], new_size[0]), interpolation=cv2.INTER_AREA)
    
    # Calculate padding to add
    delta_w = desired_size - new_size[1]
    delta_h = desired_size - new_size[0]
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)
    
    # Add black borders
    new_img = cv2.copyMakeBorder(resized, top, bottom, left, right,
                                 cv2.BORDER_CONSTANT, value=(0, 0, 0))
    return new_img

# Function to ensure each category has exactly 250 images
def ensure_250_images_per_category(dataset_dir, target_size=250):
    for category in TRASHNET_CATEGORIES:
        cat_dir = os.path.join(dataset_dir, category)
        images = os.listdir(cat_dir)
        
        # Only process if the category has more than the target size
        if len(images) > target_size:
            images = random.sample(images, target_size)
        
        # Now we copy the images to make sure each category has 250 images
        for img in images:
            src = os.path.join(cat_dir, img)
            dst = os.path.join(cat_dir, img)
            
            # Only copy if the source and destination are different
            if os.path.exists(src) and src != dst:
                shutil.copy(src, dst)  # Copy image to the same place

        # Update category count
        category_counts[category] = len(images)

# Load the annotations file
with open(ANNOTATIONS_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

images_info = data.get("images", [])
annotations_info = data.get("annotations", [])
categories_info = data.get("categories", [])

# Build category details and image-to-objects mapping
cat_id_to_details = {}
for cat in categories_info:
    cat_id = cat["id"]
    cat_name = cat["name"]
    cat_super = cat["supercategory"]
    cat_id_to_details[cat_id] = (cat_name, cat_super)

image_id_to_objects = defaultdict(list)
for ann in annotations_info:
    image_id = ann["image_id"]
    cat_id = ann["category_id"]
    bbox = ann["bbox"]
    image_id_to_objects[image_id].append((cat_id, bbox))

print("Processing TACO images and extracting cropped objects...")

# Process images
count_total = 0
count_saved = 0

for img_info in images_info:
    image_id = img_info["id"]
    file_name = img_info["file_name"]
    img_path = os.path.join(TACO_DATA_PATH, file_name)
    if not os.path.exists(img_path):
        continue
    img = cv2.imread(img_path)
    if img is None:
        continue

    annotations = image_id_to_objects.get(image_id, [])
    for i, (cat_id, bbox) in enumerate(annotations):
        count_total += 1
        x, y, w, h = bbox
        x1 = int(x)
        y1 = int(y)
        x2 = int(x + w)
        y2 = int(y + h)
        if x2 <= x1 or y2 <= y1:
            continue
        crop = img[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        crop_processed = resize_keep_aspect(crop, desired_size=128)
        
        cat_name, cat_super = cat_id_to_details.get(cat_id, ("", ""))
        trashnet_cat = map_category_to_trashnet(cat_name, cat_super)
        
        category_counts[trashnet_cat] += 1
        
        out_dir = os.path.join(OUTPUT_DATASET_DIR, trashnet_cat)
        os.makedirs(out_dir, exist_ok=True)
        out_filename = f"{image_id}_{i}.jpg"
        out_path = os.path.join(out_dir, out_filename)
        cv2.imwrite(out_path, crop_processed)
        count_saved += 1

print(f"Processed {len(images_info)} images, extracted {count_total} objects, saved {count_saved} cropped images.")

# Ensure each category has 250 images
ensure_250_images_per_category(OUTPUT_DATASET_DIR)

# Plot the cropped images per category
categories = list(category_counts.keys())
counts = list(category_counts.values())
plt.figure(figsize=(8, 6))
plt.bar(categories, counts, color='skyblue')
plt.xlabel("Category")
plt.ylabel("Number of Cropped Images")
plt.title("Cropped Images per Category")
plt.show()
