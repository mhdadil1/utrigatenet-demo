"""One-patch smoke-test training. Not a research training run."""
from pathlib import Path
import numpy as np
import tensorflow as tf
from models.utrigatenet import build_utrigate_net
from losses.urwl import urwl_loss

DATA = Path("data/processed/patient_sample.npz")
WEIGHTS = Path("models/utrigate_trained_prototype.weights.h5")

def main():
    if not DATA.exists():
        raise FileNotFoundError(f"{DATA} not found. Run python -m preprocessing.prepare_patient first.")
    sample = np.load(DATA)
    x = sample["image"][None, ...].astype(np.float32)
    y = sample["label"][None, ...].astype(np.int32)
    print("Image batch:", x.shape, "label batch:", y.shape)
    model = build_utrigate_net(input_shape=x.shape[1:])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=5e-4), loss=urwl_loss)
    model.fit(x, y, epochs=1, batch_size=1, verbose=1)
    model.save_weights(WEIGHTS)
    print("Saved weights:", WEIGHTS.resolve())

if __name__ == "__main__":
    main()
