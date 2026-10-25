# Experimental proposed extension
This folder sketches the proposed reliability-aware extension:
- training-only modality corruption and dropout;
- modality-specific TCE-lite features;
- UMRE-style learned scalar reliability per modality;
- reliability-conditioned FiLM;
- reliability-weighted fusion;
- segmentation and voxel-wise uncertainty heads.

It is a research prototype. The reliability score is learned from features but has no explicit quality-label supervision yet; the uncertainty output also needs a defined target/loss and calibration evaluation. Do not present it as a validated result. `augmentation.corrupt_modalities` returns `(corrupted_images, availability_mask)` during training.
