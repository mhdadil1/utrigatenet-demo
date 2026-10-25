from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from models.utrigatenet import build_utrigate_net

DATA = Path("data/processed/patient_sample.npz")
WEIGHTS = Path("models/utrigate_trained_prototype.weights.h5")
OUT = Path("outputs/prediction_comparison.png")

def main():
    sample = np.load(DATA)
    x = sample["image"][None, ...].astype(np.float32)
    gt = sample["label"]
    model = build_utrigate_net(input_shape=x.shape[1:])
    model.load_weights(WEIGHTS)
    probabilities = model.predict(x, verbose=0)[0]
    pred = probabilities.argmax(axis=-1)
    z = gt.shape[2] // 2
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    ax[0].imshow(x[0, :, :, z, 0].T, cmap="gray", origin="lower")
    ax[0].set_title("FLAIR")
    ax[1].imshow(gt[:, :, z].T, cmap="viridis", origin="lower", vmin=0, vmax=3)
    ax[1].set_title("Ground truth")
    ax[2].imshow(pred[:, :, z].T, cmap="viridis", origin="lower", vmin=0, vmax=3)
    ax[2].set_title("Prediction")
    for a in ax: a.axis("off")
    fig.tight_layout()
    fig.savefig(OUT, dpi=160)
    plt.close(fig)
    print("Saved:", OUT.resolve())

if __name__ == "__main__":
    main()
