"""
flatten_dataset.py
------------------
Moves all images from nested subfolders inside dataset/genuine/
into the root of dataset/genuine/ in one shot.

Handles duplicate filenames automatically by prefixing the
parent folder name — so no images are overwritten.

Usage:
    python flatten_dataset.py
"""

import os
import shutil

GENUINE_DIR = "dataset/genuine"
VALID_EXTS  = {".jpg", ".jpeg", ".png", ".webp", ".avif"}

def flatten(genuine_dir: str):
    print(f"\n📂  Scanning '{genuine_dir}' for subfolders...\n")

    moved   = 0
    skipped = 0

    for root, dirs, files in os.walk(genuine_dir):
        # Skip the root itself — only process subfolders
        if root == genuine_dir:
            continue

        folder_name = os.path.basename(root)

        for filename in files:
            ext = os.path.splitext(filename)[1].lower()
            if ext not in VALID_EXTS:
                continue

            src  = os.path.join(root, filename)

            # Prefix folder name to avoid duplicate filenames
            # e.g.  Panadol/front.jpg  →  panadol_front.jpg
            new_name = f"{folder_name.lower()}_{filename}"
            dst  = os.path.join(genuine_dir, new_name)

            # If name still clashes, add a counter
            counter = 1
            while os.path.exists(dst):
                stem, ext2 = os.path.splitext(new_name)
                dst = os.path.join(genuine_dir, f"{stem}_{counter}{ext2}")
                counter += 1

            shutil.move(src, dst)
            print(f"   ✔  {os.path.relpath(src)} → {new_name}")
            moved += 1

    # Remove now-empty subfolders
    for root, dirs, files in os.walk(genuine_dir, topdown=False):
        if root == genuine_dir:
            continue
        try:
            os.rmdir(root)
        except OSError:
            pass  # folder not empty, leave it

    print(f"\n✅  Done.")
    print(f"   Images moved : {moved}")
    print(f"\n   All images are now flat inside '{genuine_dir}/'")
    print(f"   Ready to run:  python generate_fakes.py\n")

if __name__ == "__main__":
    flatten(GENUINE_DIR)
