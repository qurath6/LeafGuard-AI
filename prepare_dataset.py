
import os
import random
import shutil

# Original dataset location
DATASET_PATH = os.path.join(
    "New Plant Diseases Dataset(Augmented)",
    "New Plant Diseases Dataset(Augmented)",
    "train"
)

# Smaller dataset location
OUTPUT_PATH = "dataset_small"

# Number of images selected from each class
IMAGES_PER_CLASS = 150

# Plant classes to use
SELECTED_PLANTS = ["Tomato", "Potato", "Grape"]

random.seed(42)

os.makedirs(OUTPUT_PATH, exist_ok=True)

if not os.path.exists(DATASET_PATH):
    print("ERROR: Training dataset folder not found!")
    raise SystemExit

class_folders = [
    folder for folder in os.listdir(DATASET_PATH)
    if os.path.isdir(os.path.join(DATASET_PATH, folder))
    and any(folder.startswith(plant + "___") for plant in SELECTED_PLANTS)
]

print("Preparing smaller dataset...")

total_images = 0

for class_name in sorted(class_folders):

    source_folder = os.path.join(DATASET_PATH, class_name)
    destination_folder = os.path.join(OUTPUT_PATH, class_name)

    os.makedirs(destination_folder, exist_ok=True)

    images = [
        file for file in os.listdir(source_folder)
        if file.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    random.shuffle(images)
    selected_images = images[:IMAGES_PER_CLASS]

    for image in selected_images:
        shutil.copy2(
            os.path.join(source_folder, image),
            os.path.join(destination_folder, image)
        )

    total_images += len(selected_images)

    print(f"{class_name}: {len(selected_images)} images copied")

print("\nDataset preparation completed!")
print("Total classes:", len(class_folders))
print("Total images:", total_images)
print("Saved inside:", OUTPUT_PATH)
