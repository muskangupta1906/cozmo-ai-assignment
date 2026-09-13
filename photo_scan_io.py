"""
photo_scan_io.py
-----------------
Loader for a photo-tier room capture: a folder of 2-8 stills, no depth, no
poses (Part 1's "floor" tier -- any iPhone 15+, any picture in).

Unlike scan_io.py (Stray Scanner LiDAR format), there is no camera_matrix.csv
and no odometry.csv here -- intrinsics and poses both have to be recovered
from the images themselves. This module only handles the intrinsics half;
pose recovery (relative camera registration between stills) is
reconstruct_room_photo.py's job, since it needs the depth model's output to
do it.
"""

import os
import glob
import math
import numpy as np
from PIL import Image, ExifTags

try:
    import pillow_heif
    pillow_heif.register_heif_opener()
except ImportError:
    # iPhone stills are HEIC by default. Without this, HEIC files simply
    # fail to load -- fine on a machine where photos were pre-converted to
    # JPEG, not fine on a raw AirDrop/iCloud folder. Documented in
    # docs/device_matrix.md rather than silently degrading.
    pass

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".heic", ".heif")

# iPhone main/wide camera horizontal FOV when no EXIF focal length is
# present -- ~26mm-equivalent, a commonly cited spec across iPhone 15+
# models. This is a fallback assumption, not a measurement: it biases
# every downstream metric distance by however far the true device/lens
# differs from it, which is exactly why photo tier gets a much wider CI
# than LiDAR (see reconstruct_room_photo.compute_confidence_intervals_photo).
FALLBACK_HFOV_DEG = 69.4


def list_photos(room_dir: str):
    paths = []
    for ext in IMAGE_EXTS:
        paths.extend(glob.glob(os.path.join(room_dir, f"*{ext}")))
        paths.extend(glob.glob(os.path.join(room_dir, f"*{ext.upper()}")))
    paths = sorted(set(paths))
    if len(paths) < 2:
        raise RuntimeError(
            f"{room_dir}: found {len(paths)} photo(s), need at least 2 to "
            f"register any pair against each other"
        )
    if len(paths) > 8:
        print(f"      NOTE: {len(paths)} photos found in {room_dir}, "
              f"capture protocol caps this tier at 8 -- using all of them anyway")
    return paths


def load_image_rgb(path: str) -> np.ndarray:
    img = Image.open(path)
    img = img.convert("RGB")
    # EXIF orientation isn't auto-applied by PIL's raw pixel array -- rotate
    # to upright first, otherwise back-projection geometry silently assumes
    # the wrong image axes.
    try:
        exif = img.getexif()
        orientation_tag = next(k for k, v in ExifTags.TAGS.items() if v == "Orientation")
        orientation = exif.get(orientation_tag)
        if orientation == 3:
            img = img.rotate(180, expand=True)
        elif orientation == 6:
            img = img.rotate(-90, expand=True)
        elif orientation == 8:
            img = img.rotate(90, expand=True)
    except (StopIteration, KeyError, AttributeError):
        pass
    return np.array(img)


def get_intrinsics(path: str, image_shape) -> tuple:
    """
    Returns (fx, fy, cx, cy, exif_used: bool) in pixels for an image of the
    given (h, w[, c]) shape.

    Principal point is assumed at the image center in both cases -- a
    standard approximation absent a real calibration, and a much smaller
    source of error than the focal length itself.
    """
    h, w = image_shape[0], image_shape[1]
    cx, cy = w / 2.0, h / 2.0

    focal_35mm = None
    try:
        img = Image.open(path)
        exif = img.getexif()
        tag_map = {v: k for k, v in ExifTags.TAGS.items()}
        fl35_tag = tag_map.get("FocalLengthIn35mmFilm")
        if fl35_tag is not None and fl35_tag in exif:
            focal_35mm = float(exif[fl35_tag])
    except Exception:
        focal_35mm = None

    if focal_35mm and focal_35mm > 0:
        # Standard 35mm-equivalent conversion: the "35mm equivalent" focal
        # length is defined against a 36mm-wide full-frame sensor, so the
        # horizontal FOV is recoverable regardless of the phone's actual
        # (much smaller) sensor size.
        hfov = 2 * math.atan(36.0 / (2 * focal_35mm))
        fx = (w / 2.0) / math.tan(hfov / 2.0)
        fy = fx  # square pixels assumed, true for phone cameras
        return fx, fy, cx, cy, True

    hfov = math.radians(FALLBACK_HFOV_DEG)
    fx = (w / 2.0) / math.tan(hfov / 2.0)
    fy = fx
    return fx, fy, cx, cy, False


def load_room_photos(room_dir: str):
    """Returns a list of dicts: {path, image (HxWx3 uint8), fx, fy, cx, cy,
    exif_used}, one per photo, sorted by filename (assumed capture order)."""
    out = []
    for p in list_photos(room_dir):
        img = load_image_rgb(p)
        fx, fy, cx, cy, exif_used = get_intrinsics(p, img.shape)
        out.append({"path": p, "image": img, "fx": fx, "fy": fy,
                    "cx": cx, "cy": cy, "exif_used": exif_used})
    n_exif = sum(1 for o in out if o["exif_used"])
    print(f"      loaded {len(out)} photos from {room_dir} "
          f"({n_exif}/{len(out)} with EXIF focal length, "
          f"{len(out) - n_exif} on the {FALLBACK_HFOV_DEG} deg fallback FOV)")
    return out
