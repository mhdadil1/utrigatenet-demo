"""Evaluate a saved prototype on the same sample patch (smoke test only)."""
from pathlib import Path
import numpy as np
import tensorflow as tf
from models.utrigatenet import build_utrigate_net
from losses.urwl import urwl_loss

DATA = Path("data/processed/patient_sample.npz")
WEIGHTS = Path("models/utrigate_trained_prototype.weights.h5")

def dice_for_class(gt, pred, class_id, eps=1e-7):
    a, b = gt == class_id, pred == class_id
    return float((2 * np.logical_and(a, b).sum() + eps) /
                 (a.sum() + b.sum() + eps))

def main():
    sample = np.load(DATA)
    x = sample["image"][None, ...].astype(np.float32)
    gt = sample["label"].astype(np.uint8)
    model = build_utrigate_net(input_shape=x.shape[1:])
    model.compile(optimizer=tf.keras.optimizers.Adam(5e-4), loss=urwl_loss)
    if not WEIGHTS.exists():
        raise FileNotFoundError(f"{WEIGHTS} missing. Run python -m training.train first.")
    model.load_weights(WEIGHTS)
    pred = np.argmax(model.predict(x, verbose=0)[0], axis=-1).astype(np.uint8)
    print("MRI input shape:", x.shape)
    print("Ground-truth shape:", gt.shape)
    print("Ground-truth classes:", np.unique(gt))
    print("Predicted classes:", np.unique(pred))
    for cls in (1, 2, 3):
        print(f"Dice class {cls}: {dice_for_class(gt, pred, cls):.4f}")
    print("Warning: this is the training sample, not an independent test set.")

if __name__ == "__main__":
    main()
