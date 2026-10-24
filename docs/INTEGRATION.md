# Framework integration

This package contains only the HLCQ-DETR-specific source. It is intended to be
applied to the authors' Ultralytics-based RT-DETR framework (version 8.0.201).

1. Copy `framework_patch/ultralytics/nn/extra_modules/hlcq_modules.py` into
   `ultralytics/nn/extra_modules/`.
2. Add the following import to `ultralytics/nn/extra_modules/__init__.py`:

   ```python
   from .hlcq_modules import *
   ```

3. In `ultralytics/nn/tasks.py`, locate the module tuple in `parse_model` that
   prepends `c1` and `c2` to module arguments. Add both names to that tuple:

   ```python
   HighFrequencyPreservingDownsample,
   LocalContrastNeighborhoodConsensus,
   ```

4. Copy `framework_patch/ultralytics/cfg/models/rt-detr/hlcq-detr-r18.yaml`
   into the framework's RT-DETR configuration directory, or keep the path used
   by `scripts/train.py`.
5. Download the dataset from the Kaggle URL in `README.md`, extract it into the
   repository-relative `dataset/` directory, and organize the images and labels
   according to the documented layout. The included `data.yaml` already points
   to this location. If a different dataset is used, copy `data.example.yaml`,
   edit its path and class names, and save it as `data.yaml`.
6. Run `python scripts/train.py` from the repository root.

The CP2Q component is represented directly by the compact P2 branch and the
four-input `RTDETRDecoder` declaration in the model YAML; it does not require a
separate Python module.
