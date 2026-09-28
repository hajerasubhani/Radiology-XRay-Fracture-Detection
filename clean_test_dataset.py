from PIL import Image
from pathlib import Path
import shutil

source_folder = Path("test")
clean_folder = Path("test_clean")

# Create clean folders
for class_folder in ["fractured", "not fractured"]:
    (clean_folder / class_folder).mkdir(parents=True, exist_ok=True)

total_images = 0
copied_images = 0
skipped_images = 0

print("Creating clean test dataset...\n")

for class_folder in ["fractured", "not fractured"]:

    source_class_folder = source_folder / class_folder
    clean_class_folder = clean_folder / class_folder

    for image_path in source_class_folder.iterdir():

        if image_path.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
            continue

        total_images += 1

        try:
            # Try to fully load the image
            with Image.open(image_path) as img:
                img.load()

            # If successful, copy it
            shutil.copy2(
                image_path,
                clean_class_folder / image_path.name
            )

            copied_images += 1

        except Exception as e:

            skipped_images += 1

            print("SKIPPED:")
            print(image_path)
            print("Reason:", e)
            print("--------------------------------")

print("\n======================================")
print("CLEAN DATASET CREATED")
print("======================================")

print(f"Total images found: {total_images}")
print(f"Images copied: {copied_images}")
print(f"Images skipped: {skipped_images}")

print("\nClean dataset location:")
print(clean_folder)