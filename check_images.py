from PIL import Image
from pathlib import Path

folders = ["train", "val", "test"]

bad_images = []

for folder in folders:
    folder_path = Path(folder)

    print(f"\nChecking: {folder}")

    for image_path in folder_path.rglob("*"):

        if image_path.suffix.lower() in [".jpg", ".jpeg", ".png"]:

            try:
                with Image.open(image_path) as img:
                    img.verify()

            except Exception as e:
                print(f"BAD IMAGE: {image_path}")
                print(f"ERROR: {e}")

                bad_images.append(str(image_path))

print("\n--------------------------------")
print(f"Total corrupted images: {len(bad_images)}")
print("--------------------------------")

if bad_images:
    with open("bad_images.txt", "w") as file:
        for image in bad_images:
            file.write(image + "\n")

    print("Bad image list saved to bad_images.txt")

else:
    print("All images are valid!")