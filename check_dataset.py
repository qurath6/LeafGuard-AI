
import os

# Dataset location
dataset_path = "New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)"

train_path = os.path.join(dataset_path, "train")

print("\nLEAFGUARD AI - DATASET CHECK")
print("-" * 40)

if not os.path.exists(train_path):
    print("Training folder not found!")
else:
    classes = sorted(os.listdir(train_path))

    print("Total classes found:", len(classes))
    print("\nAvailable plant disease classes:")

    for class_name in classes:
        class_path = os.path.join(train_path, class_name)

        if os.path.isdir(class_path):
            image_count = len([
                file for file in os.listdir(class_path)
                if file.lower().endswith((".jpg", ".jpeg", ".png"))
            ])

            print(f"{class_name}: {image_count} images")

print("\nDataset check completed!")

valid_path = os.path.join(dataset_path, "valid")

if os.path.exists(valid_path):
    print("\nValidation folder found!")
    print("Validation classes:", len(os.listdir(valid_path)))
else:
    print("\nValidation folder not found!")