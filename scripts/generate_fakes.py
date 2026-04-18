"""
generate_fakes.py
-----------------
Reads every image from  dataset/genuine/
Applies controlled distortions that mimic real-world counterfeit
packaging imperfections, then saves results to  dataset/fake/

Distortions applied (individually and in combination):
  1. Logo blur          – simulates low-resolution printing
  2. Font colour shift  – ink tone variation from cheaper inks
  3. Geometric skew     – slight perspective warp (misaligned print run)
  4. Brightness noise   – uneven lighting from poor press calibration
  5. Edge degradation   – border thickness inconsistency
  6. JPEG compression   – reproduction-quality loss artefacts
  7. Colour channel swap– subtle hue inversion in one channel
  8. Synthetic label text – bright red EXP/BATCH lines mimicking pasted or
     digitally added text on real boxes (common counterfeit / tamper cue)

Each genuine image produces N_FAKES_PER_IMAGE fake variant(s). For N_FAKES=1,
each output image is either (a) classic distortions plus synthetic EXP/BATCH
text, or (b) tamper-focused (mostly overlay, like “same photo + red expiry”),
chosen at random so both cues are seen without doubling the fake class size.

Usage:
    pip install Pillow numpy opencv-python
    python generate_fakes.py
"""

import os
import random
import argparse
import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFont, ImageOps
import cv2

# ── Config ────────────────────────────────────────────────────────────────────
GENUINE_DIR     = "dataset/genuine"
FAKE_DIR        = "dataset/fake"
N_FAKES_PER_IMAGE = 1       # One synthetic fake per genuine; type alternates (see loop)
TARGET_SIZE       = (224, 224)  # Must match model input size
SEED              = 42
VALID_EXTS        = {".jpg", ".jpeg", ".png", ".webp"}

random.seed(SEED)
np.random.seed(SEED)


# ── Individual distortion functions ──────────────────────────────────────────

def apply_logo_blur(img: Image.Image) -> Image.Image:
    """Simulate low-res logo printing via Gaussian blur on a random region."""
    arr = np.array(img)
    h, w = arr.shape[:2]
    # Pick a random rectangular region (upper-left quadrant likely has logo)
    x1 = random.randint(0, w // 3)
    y1 = random.randint(0, h // 3)
    x2 = random.randint(w // 3, 2 * w // 3)
    y2 = random.randint(h // 3, 2 * h // 3)

    region = arr[y1:y2, x1:x2]
    blurred = cv2.GaussianBlur(region, (15, 15), sigmaX=random.uniform(2, 5))
    arr[y1:y2, x1:x2] = blurred
    return Image.fromarray(arr)


def apply_colour_shift(img: Image.Image) -> Image.Image:
    """Shift hue/saturation to mimic inferior ink quality."""
    arr = np.array(img.convert("HSV") if hasattr(img, 'convert') else img)
    # Work in RGB; shift red channel slightly
    arr = np.array(img)
    channel = random.randint(0, 2)
    shift   = random.randint(15, 40) * random.choice([-1, 1])
    arr[:, :, channel] = np.clip(arr[:, :, channel].astype(int) + shift, 0, 255)
    return Image.fromarray(arr.astype(np.uint8))


def apply_geometric_skew(img: Image.Image) -> Image.Image:
    """Apply a subtle perspective warp to simulate misaligned printing."""
    arr = np.array(img)
    h, w = arr.shape[:2]
    jitter = lambda: random.randint(-12, 12)
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = np.float32([
        [jitter(), jitter()],
        [w + jitter(), jitter()],
        [w + jitter(), h + jitter()],
        [jitter(), h + jitter()]
    ])
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(arr, M, (w, h),
                                  borderMode=cv2.BORDER_REPLICATE)
    return Image.fromarray(warped)


def apply_brightness_noise(img: Image.Image) -> Image.Image:
    """Add random brightness variation and slight noise."""
    # Brightness
    factor = random.uniform(0.65, 1.35)
    img = ImageEnhance.Brightness(img).enhance(factor)
    # Gaussian noise
    arr = np.array(img).astype(np.int16)
    noise = np.random.normal(0, random.uniform(5, 18), arr.shape)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def apply_edge_degradation(img: Image.Image) -> Image.Image:
    """Erode borders/edges to simulate cheap cutting and lamination."""
    arr = np.array(img)
    kernel_size = random.choice([3, 5])
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    degraded = cv2.erode(arr, kernel, iterations=1)
    # Blend with original so effect is subtle
    blended = cv2.addWeighted(arr, 0.6, degraded, 0.4, 0)
    return Image.fromarray(blended)


def apply_jpeg_compression(img: Image.Image) -> Image.Image:
    """Re-encode with low quality to introduce compression artefacts."""
    import io
    quality = random.randint(30, 60)
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    return Image.open(buffer).copy()


def apply_channel_inversion(img: Image.Image) -> Image.Image:
    """Invert one colour channel slightly (subtle hue anomaly)."""
    arr   = np.array(img)
    ch    = random.randint(0, 2)
    alpha = random.uniform(0.08, 0.22)   # How strong the inversion is
    inv   = 255 - arr[:, :, ch]
    arr[:, :, ch] = np.clip(
        arr[:, :, ch] * (1 - alpha) + inv * alpha, 0, 255
    ).astype(np.uint8)
    return Image.fromarray(arr)


def _load_sans_font(size: int):
    """Best-effort readable sans font for synthetic stamp text."""
    candidates = [
        os.path.join(os.environ.get("WINDIR", ""), "Fonts", "arial.ttf"),
        os.path.join(os.environ.get("WINDIR", ""), "Fonts", "calibri.ttf"),
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ]
    for path in candidates:
        if path and os.path.isfile(path):
            try:
                return ImageFont.truetype(path, size=max(10, size))
            except OSError:
                continue
    return ImageFont.load_default()


def apply_expiry_text_overlay(img: Image.Image) -> Image.Image:
    """
    Simulate pasted / digitally added expiry or batch lines (high-contrast red),
    often misaligned with original print — a common counterfeit cue.
    """
    out = img.copy()
    w, h = out.size
    draw = ImageDraw.Draw(out)
    font_size = max(12, min(h // 14, w // 12))
    font = _load_sans_font(font_size)

    month = random.randint(1, 12)
    year = random.choice([1998, 1999, 2000, 2001, 2002, 2003])
    exp_line = f"EXP: {month:02d}/{year}"
    batch_line = f"BATCH: {random.randint(10000, 99999)}"

    # Lower band where expiry often appears; randomize so the model generalizes
    x0 = random.randint(max(4, w // 20), w // 2)
    y0 = random.randint(int(h * 0.52), int(h * 0.82))
    red = (random.randint(200, 255), random.randint(10, 60), random.randint(20, 70))

    for dy, line in enumerate((exp_line, batch_line)):
        y = y0 + dy * (font_size + random.randint(2, 8))
        if y >= h - 4:
            break
        # Slight outline so text stays visible on varied backgrounds
        for dx, dy2 in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            draw.text((x0 + dx, y + dy2), line, fill=(20, 20, 20), font=font)
        draw.text((x0, y), line, fill=red, font=font)

    return out


# ── Distortion registry ───────────────────────────────────────────────────────
ALL_DISTORTIONS = [
    apply_logo_blur,
    apply_colour_shift,
    apply_geometric_skew,
    apply_brightness_noise,
    apply_edge_degradation,
    apply_jpeg_compression,
    apply_channel_inversion,
]


def make_fake_distortion_stack(img: Image.Image, n_distortions: int = None) -> Image.Image:
    """
    Apply a random combination of classic packaging distortions (no text overlay).
    """
    if n_distortions is None:
        n_distortions = random.randint(2, 4)

    chosen = random.sample(ALL_DISTORTIONS, min(n_distortions, len(ALL_DISTORTIONS)))
    result = img.copy()
    for fn in chosen:
        try:
            result = fn(result)
        except Exception as e:
            print(f"   ⚠  Distortion {fn.__name__} skipped: {e}")
    return result


def make_fake_distortion_plus_overlay(img: Image.Image) -> Image.Image:
    """Classic counterfeit artefacts + synthetic label lines (common judge/demo case)."""
    result = make_fake_distortion_stack(img)
    try:
        result = apply_expiry_text_overlay(result)
    except Exception as e:
        print(f"   ⚠  Expiry overlay skipped: {e}")
    return result


def make_fake_tamper_overlay(img: Image.Image) -> Image.Image:
    """
    Same packaging photo with digitally added expiry/batch text — minimal global blur/skew.
    Matches “photograph unchanged + red EXP added” tampering.
    """
    result = img.copy()
    try:
        result = apply_expiry_text_overlay(result)
    except Exception as e:
        print(f"   ⚠  Expiry overlay skipped: {e}")
        return make_fake_distortion_stack(img)
    if random.random() < 0.45:
        try:
            result = apply_brightness_noise(result)
        except Exception:
            pass
    if random.random() < 0.3:
        try:
            result = apply_jpeg_compression(result)
        except Exception:
            pass
    return result


def make_fake(img: Image.Image, n_distortions: int = None) -> Image.Image:
    """Backward-compatible: distortions then expiry overlay."""
    del n_distortions  # unused; kept for API compatibility
    return make_fake_distortion_plus_overlay(img)


# ── Main pipeline ─────────────────────────────────────────────────────────────

def generate(genuine_dir: str, fake_dir: str, n_fakes: int):
    os.makedirs(fake_dir, exist_ok=True)

    genuine_files = [
        f for f in os.listdir(genuine_dir)
        if os.path.splitext(f)[1].lower() in VALID_EXTS
    ]

    if not genuine_files:
        print(f"\n❌  No images found in '{genuine_dir}'.")
        print("    Add .jpg / .png genuine medicine photos first.\n")
        return

    print(f"\n🔬  Found {len(genuine_files)} genuine images.")
    print(f"    Generating {n_fakes} fake variant(s) per image ...\n")

    total_generated = 0
    for idx, filename in enumerate(genuine_files, 1):
        src_path = os.path.join(genuine_dir, filename)
        try:
            img = Image.open(src_path).convert("RGB")
            img = img.resize(TARGET_SIZE, Image.LANCZOS)
        except Exception as e:
            print(f"   ⚠  Skipping {filename}: {e}")
            continue

        stem, _ = os.path.splitext(filename)

        for variant in range(1, n_fakes + 1):
            if n_fakes >= 2:
                if variant >= 2:
                    fake_img = make_fake_tamper_overlay(img)
                    tag = f"v{variant:02d}_tamper"
                else:
                    fake_img = make_fake_distortion_plus_overlay(img)
                    tag = f"v{variant:02d}"
            else:
                if random.random() < 0.5:
                    fake_img = make_fake_tamper_overlay(img)
                    tag = "v01_tamper"
                else:
                    fake_img = make_fake_distortion_plus_overlay(img)
                    tag = "v01"
            out_name = f"fake_{stem}_{tag}.jpg"
            out_path = os.path.join(fake_dir, out_name)
            fake_img.save(out_path, "JPEG", quality=92)
            total_generated += 1

        if idx % 20 == 0 or idx == len(genuine_files):
            print(f"   [{idx}/{len(genuine_files)}] processed ...")

    print(f"\n✅  Done. {total_generated} fake images saved to '{fake_dir}/'")
    print(f"\n   Genuine : {len(genuine_files):>5} images")
    print(f"   Fake    : {total_generated:>5} images")

    if abs(len(genuine_files) - total_generated) > len(genuine_files) * 0.2:
        print("\n   ⚠  Class imbalance detected.")
        print("      Adjust N_FAKES_PER_IMAGE at the top of this file to balance classes.")

    print("\n   Ready to train → run:  python train.py\n")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate simulated counterfeit medicine images."
    )
    parser.add_argument("--genuine", default=GENUINE_DIR,
                        help="Path to genuine images folder")
    parser.add_argument("--fake",    default=FAKE_DIR,
                        help="Path to output fake images folder")
    parser.add_argument("--n",       type=int, default=N_FAKES_PER_IMAGE,
                        help="Number of fake variants per genuine image")
    args = parser.parse_args()

    generate(args.genuine, args.fake, args.n)
