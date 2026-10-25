"""Extract one tumour-centred 32^3 patch from one BraTS 2019 patient."""
from pathlib import Path
import numpy as np
import nibabel as nib

BRATS_ROOT = Path(r"C:\Users\GOD\Brats 2019\MICCAI_BraTS_2019_Data_Training")
OUTPUT_PATH = Path("data/processed/patient_sample.npz")
PATCH_SIZE = 32

def find_case(root: Path):
    for grade in ("HGG", "LGG"):
        folder = root / grade
        if folder.exists():
            cases = sorted(p for p in folder.iterdir() if p.is_dir())
            if cases:
                return cases[0]
    raise FileNotFoundError(f"No HGG/LGG case folders found under {root}")

def load_volume(path: Path):
    return np.asarray(nib.load(str(path)).get_fdata(), dtype=np.float32)

def normalize_nonzero(volume):
    mask = volume != 0
    out = np.zeros_like(volume, dtype=np.float32)
    if not np.any(mask):
        return out
    vals = volume[mask]
    mean, std = float(vals.mean()), float(vals.std())
    if std < 1e-6:
        out[mask] = vals - mean
    else:
        out[mask] = (vals - mean) / std
    return out

def crop_center(volume, center, size=PATCH_SIZE):
    starts = [int(c) - size // 2 for c in center]
    result = np.zeros((size, size, size), dtype=volume.dtype)
    src_slices, dst_slices = [], []
    for axis, start in enumerate(starts):
        end = start + size
        src_start, src_end = max(0, start), min(volume.shape[axis], end)
        dst_start = max(0, -start)
        dst_end = dst_start + max(0, src_end - src_start)
        src_slices.append(slice(src_start, src_end))
        dst_slices.append(slice(dst_start, dst_end))
    result[tuple(dst_slices)] = volume[tuple(src_slices)]
    return result

def main():
    case = find_case(BRATS_ROOT)
    print("Using case:", case)
    names = ("flair", "t1", "t1ce", "t2")
    volumes = []
    for name in names:
        path = case / f"{case.name}_{name}.nii"
        if not path.exists():
            # Some datasets use .nii.gz
            path = case / f"{case.name}_{name}.nii.gz"
        if not path.exists():
            raise FileNotFoundError(f"Missing modality file: {path}")
        volumes.append(normalize_nonzero(load_volume(path)))

    seg_path = case / f"{case.name}_seg.nii"
    if not seg_path.exists():
        seg_path = case / f"{case.name}_seg.nii.gz"
    if not seg_path.exists():
        raise FileNotFoundError(f"Missing segmentation file: {seg_path}")
    seg = load_volume(seg_path).astype(np.uint8)
    seg[seg == 4] = 3

    tumour = np.argwhere(seg > 0)
    center = tumour.mean(axis=0).astype(int) if len(tumour) else np.array(seg.shape) // 2
    image_patch = np.stack([crop_center(v, center) for v in volumes], axis=-1)
    label_patch = crop_center(seg, center).astype(np.uint8)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUTPUT_PATH, image=image_patch.astype(np.float32), label=label_patch)
    print("Saved:", OUTPUT_PATH.resolve())
    print("Image shape:", image_patch.shape, "label shape:", label_patch.shape)
    print("Labels:", np.unique(label_patch))

if __name__ == "__main__":
    main()
