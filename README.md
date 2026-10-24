# HLCQ-DETR

This repository provides the core implementation and reproducibility materials
for HLCQ-DETR, an RT-DETR-based object detector combining Compact P2 Query
(CP2Q), High-Frequency-Preserving Downsampling (HFPD), and Local Contrast
Neighborhood Consensus (LCNC).

## Available Materials

- the final HLCQ-DETR model configuration (CP2Q + HFPD + LCNC);
- implementations of the HFPD and LCNC modules;
- the CP2Q four-scale detection architecture defined in the model configuration;
- sanitized training and validation entry points;
- relative-path dataset configuration and framework integration instructions.

## Dataset

The dataset files are not included in this repository. Download the publicly
available [Kidney Stone Images with Bounding Box Annotations](https://www.kaggle.com/datasets/safurahajiheidari/kidney-stone-images)
dataset from Kaggle before training.

The dataset was created by Safura Hajiheidari and is distributed under the
[Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/)
license. Please cite the original dataset and comply with its license terms.

The dataset contains one object category, `Tas_Var`, and is divided as follows:

- training set: 1,054 images and 1,054 label files;
- validation set: 123 images and 123 label files;
- test set: 123 images and 123 label files.

After downloading, arrange the files as follows:

```text
dataset/
|-- images/
|   |-- train/
|   |-- val/
|   `-- test/
`-- labels/
    |-- train/
    |-- val/
    `-- test/
```

The repository's `data.yaml` already points to this relative directory layout.

## Usage

Read `docs/INTEGRATION.md` to integrate the HLCQ-DETR modules into the
Ultralytics-based RT-DETR framework. After completing the integration, run the
following commands from the repository root:

```bash
python scripts/train.py
python scripts/val.py
```

Training and validation outputs are written to `runs/train/hlcq-detr` and
`runs/val/hlcq-detr`, respectively.
