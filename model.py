"""Experimental reliability-aware multimodal 3D segmentation model."""
import tensorflow as tf
from tensorflow.keras import layers, Model

def _conv_block(x, filters, name):
    x = layers.Conv3D(filters, 3, padding="same", activation="relu", name=f"{name}_conv1")(x)
    x = layers.Conv3D(filters, 3, padding="same", activation="relu", name=f"{name}_conv2")(x)
    return x

def _tce_lite(x, filters, name):
    branches = [
        layers.Conv3D(filters, (1,3,3), padding="same", dilation_rate=(1,2,2), activation="relu", name=f"{name}_axial")(x),
        layers.Conv3D(filters, (3,1,3), padding="same", dilation_rate=(2,1,2), activation="relu", name=f"{name}_coronal")(x),
        layers.Conv3D(filters, (3,3,1), padding="same", dilation_rate=(2,2,1), activation="relu", name=f"{name}_sagittal")(x),
    ]
    pooled = [layers.GlobalAveragePooling3D()(b) for b in branches]
    descriptor = layers.Concatenate()(pooled)
    weights = layers.Dense(3, activation="softmax", name=f"{name}_weights")(descriptor)
    weighted = []
    for i, b in enumerate(branches):
        w = layers.Lambda(lambda z, idx=i: z[:, idx:idx+1])(weights)
        w = layers.Reshape((1,1,1,1))(w)
        weighted.append(layers.Multiply()([b, w]))
    return layers.Conv3D(filters, 1, padding="same", activation="relu",
                         name=f"{name}_projection")(layers.Add()(weighted))

def build_reliability_aware_model(input_shape=(32,32,32,4), base_filters=8):
    inputs = layers.Input(input_shape, name="four_mri_modalities")
    modality_feats, reliability_values = [], []
    for i, name in enumerate(("flair", "t1", "t1ce", "t2")):
        m = layers.Lambda(lambda t, idx=i: t[..., idx:idx+1], name=f"extract_{name}")(inputs)
        f = layers.Conv3D(base_filters, 3, padding="same", activation="relu", name=f"{name}_stem")(m)
        f = _tce_lite(f, base_filters, f"{name}_tce")
        # Reliability score from feature summary. In a full system this should also use
        # explicit image-quality cues and an availability mask.
        descriptor = layers.GlobalAveragePooling3D(name=f"{name}_reliability_pool")(f)
        r = layers.Dense(1, activation="sigmoid", name=f"{name}_reliability")(descriptor)
        gamma = layers.Dense(base_filters, activation="sigmoid", name=f"{name}_film_gamma")(r)
        beta = layers.Dense(base_filters, activation="tanh", name=f"{name}_film_beta")(r)
        gamma = layers.Reshape((1,1,1,base_filters))(gamma)
        beta = layers.Reshape((1,1,1,base_filters))(beta)
        conditioned = layers.Add(name=f"{name}_film_add")([
            layers.Multiply()([f, gamma]), beta
        ])
        modality_feats.append(conditioned)
        reliability_values.append(r)

    # Reliability-weighted gated fusion; availability masking should be supplied when
    # real missing-modality metadata is available.
    reliabilities = layers.Concatenate(name="modality_reliabilities")(reliability_values)
    weights = layers.Softmax(name="reliability_fusion_weights")(reliabilities)
    weighted = []
    for i, f in enumerate(modality_feats):
        w = layers.Lambda(lambda z, idx=i: z[:, idx:idx+1])(weights)
        w = layers.Reshape((1,1,1,1))(w)
        weighted.append(layers.Multiply()([f, w]))
    fused = layers.Concatenate(name="reliability_weighted_concat")(weighted)
    fused = layers.Conv3D(base_filters, 1, padding="same", activation="relu", name="fusion_projection")(fused)

    skip = _conv_block(fused, base_filters, "encoder")
    down = layers.MaxPool3D()(skip)
    bottleneck = _conv_block(down, base_filters*2, "bottleneck")
    up = layers.UpSampling3D()(bottleneck)
    up = layers.Concatenate()([up, skip])
    decoded = _conv_block(up, base_filters, "decoder")

    # Four-class segmentation to match the current label preprocessing.
    segmentation = layers.Conv3D(4, 1, activation="softmax", name="segmentation")(decoded)
    uncertainty = layers.Conv3D(1, 1, activation="sigmoid", name="uncertainty")(decoded)
    reliability_output = layers.Lambda(lambda z: z, name="reliability_output")(reliabilities)
    return Model(inputs, [segmentation, uncertainty, reliability_output],
                 name="experimental_reliability_aware_utrigatenet")
