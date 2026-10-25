from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

DATA = Path("data/processed/patient_sample.npz")
OUT = Path("outputs/patient_visualization.png")

def main():
    sample = np.load(DATA)
    image, label = sample["image"], sample["label"]
    z = label.shape[2] // 2
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(1, 5, figsize=(15, 4))
    names = ["FLAIR", "T1", "T1ce", "T2", "Label"]
    for i in range(4):
        ax[i].imshow(image[:, :, z, i].T, cmap="gray", origin="lower")
        ax[i].set_title(names[i]); ax[i].axis("off")
    ax[4].imshow(label[:, :, z].T, cmap="viridis", origin="lower")
    ax[4].set_title(names[4]); ax[4].axis("off")
    fig.tight_layout()
    fig.savefig(OUT, dpi=160)
    plt.close(fig)
    print("Saved:", OUT.resolve())

if __name__ == "__main__":
    main()
