from PIL import Image
from pathlib import Path

test_folder = Path("test")

bad_images = []

print("Checking test images...\n")

for image_path in test_folder.rglob("*"):

    if image_path.suffix.lower() in [".jpg", ".jpeg", ".png"]:

        try:
            with Image.open(image_path) as img:

                # First check
                img.verify()

            # Second check: actually open the image
            with Image.open(image_path) as img:
                img.load()

        except Exception as e:

            print("BAD IMAGE:")
            print(image_path)
            print("ERROR:", e)
            print("--------------------------------")

            bad_images.append(str(image_path))


print("\n======================================")
print("TEST IMAGE CHECK COMPLETED")
print("======================================")

print(f"Total problematic images: {len(bad_images)}")


if bad_images:

    with open("bad_test_images.txt", "w") as file:

        for image in bad_images:
            file.write(image + "\n")

    print("\nProblematic image list saved as:")
    print("bad_test_images.txt")

else:

    print("\nAll test images passed the PIL check!")