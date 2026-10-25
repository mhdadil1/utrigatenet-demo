"""Training-only MRI modality corruption and modality dropout utilities."""
import tensorflow as tf

def corrupt_modalities(images, noise_std=0.05, contrast_range=(0.7, 1.3),
                       modality_dropout_probability=0.15, training=True):
    """images: [B,D,H,W,4], modality order FLAIR,T1,T1ce,T2."""
    if not training:
        return images
    x = tf.cast(images, tf.float32)
    batch = tf.shape(x)[0]
    # Independent contrast scaling per sample and modality.
    low, high = contrast_range
    scales = tf.random.uniform((batch, 1, 1, 1, 4), low, high)
    x = x * scales
    x = x + tf.random.normal(tf.shape(x), stddev=noise_std)
    # Drop whole modalities per sample; keep at least one modality.
    keep = tf.cast(tf.random.uniform((batch, 1, 1, 1, 4)) >= modality_dropout_probability, x.dtype)
    all_missing = tf.reduce_sum(keep, axis=-1, keepdims=True) < 1
    first_modality = tf.concat([tf.ones_like(keep[..., :1]), tf.zeros_like(keep[..., 1:])], axis=-1)
    keep = tf.where(all_missing, first_modality, keep)
    return x * keep, tf.squeeze(keep[:, 0, 0, 0, :], axis=1)
