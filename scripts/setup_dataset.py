"""
setup_dataset.py
----------------
Run this FIRST before anything else.

Creates the required folder structure for the counterfeit medicine
detection dataset and prints instructions for adding genuine images.

Usage:
    python setup_dataset.py
"""

import os

# ── Folder layout ─────────────────────────────────────────────────────────────
FOLDERS = [
    "dataset/genuine",          # Place real medicine photos here
    "dataset/fake",             # Auto-populated by generate_fakes.py
    "dataset/augmented/genuine",# Created during training pipeline
    "dataset/augmented/fake",
    "model/saved_model",        # Final exported model lands here
    "model/checkpoints",        # Best weights saved during training
    "logs/training",            # TensorBoard logs
]

def create_structure():
    base = os.path.dirname(os.path.abspath(__file__))
    print("\n📁  Creating dataset folder structure...\n")

    for folder in FOLDERS:
        path = os.path.join(base, folder)
        os.makedirs(path, exist_ok=True)
        print(f"   ✔  {folder}")

    # Drop a README inside dataset/genuine so the folder isn't empty
    readme_path = os.path.join(base, "dataset/genuine/README.txt")
    with open(readme_path, "w") as f:
        f.write(
            "GENUINE MEDICINE IMAGES\n"
            "=======================\n\n"
            "Place your genuine medicine packaging photographs here.\n\n"
            "Accepted formats : .jpg  .jpeg  .png  .webp\n"
            "Recommended count: 200–500 images to start; 1000+ for best results\n\n"
            "Sources to collect from:\n"
            "  • Tata 1mg / PharmEasy / Netmeds  (download product images)\n"
            "  • Physical photos of medicines at varied angles & lighting\n"
            "  • CDSCO / FDA drug registry reference images\n\n"
            "Naming convention:\n"
            "  brand_productname_angle_001.jpg\n"
            "  e.g.  cipla_azithral500_front_001.jpg\n\n"
            "Once images are added, run:\n"
            "  python generate_fakes.py   -> creates counterfeit samples\n"
            "  python train.py            -> trains the model\n"
        )

    print("\n✅  Structure ready.\n")
    print("=" * 58)
    print("  NEXT STEPS")
    print("=" * 58)
    print("""
  1. Add genuine medicine images to:
         dataset/genuine/

     Minimum: 200 images   |   Recommended: 500+

  2. Accepted formats : JPG, JPEG, PNG, WEBP

  3. Once images are in place, run:
         python generate_fakes.py
     This auto-generates counterfeit samples from your genuine ones.

  4. Then train the model:
         python train.py
""")

if __name__ == "__main__":
    create_structure()
