# Lunar Crater Detection (YOLOv11 + ResNet101)

A lunar crater detection pipeline built on **YOLOv11** with its default
backbone swapped for a **timm ResNet101**, trained on two Chandrayaan-2
datasets (OHRC and TMC-2).

Training runs on Google Colab (GPU required); this repo holds the notebooks,
config, and documentation needed to reproduce those runs.

> **Note:** No training is run from this repo. The notebooks in
> [`notebooks/`](notebooks/) are meant to be opened in Google Colab.

## What the project does

The pipeline detects impact craters in lunar surface imagery captured by
ISRO's Chandrayaan-2 orbiter. It trains two separate YOLOv11 object detectors
— one per camera/dataset — since OHRC and TMC-2 imagery differ enough in
resolution and crater scale that a single model handling both well is
impractical. Both models share the same architecture (yolo11.yaml) but are
trained independently with dataset-specific epoch counts.

## Architecture: why swap the backbone?

Stock YOLOv11 uses a CSPDarknet-style backbone trained from scratch (or from
COCO weights) alongside the rest of the network. This project instead plugs
in a **ResNet101 from the [timm](https://github.com/huggingface/pytorch-image-models)
library, pretrained on ImageNet**, as the feature extractor, and keeps
YOLOv11's neck (PANet-style FPN) and detection head unchanged.

The motivation, in plain terms:
- **Craters are low-contrast, texture-heavy targets.** A deeper, ImageNet-pretrained
  backbone brings feature representations already tuned to a huge variety of
  textures and edges, which transfers reasonably well to crater rims and
  shadows — useful when the labeled lunar dataset itself is small.
- **Transfer learning speeds up convergence.** Starting from ImageNet weights
  instead of random initialization typically needs fewer epochs to reach a
  usable detector, which matters when training on Colab's time-limited free
  GPUs.
- **timm gives easy access to well-tested backbone implementations** without
  having to hand-port ResNet into the Ultralytics codebase.

This required a small patch to Ultralytics itself — as of this writing, timm
backbone support isn't in a released version, only in
[PR #19609](https://github.com/ultralytics/ultralytics/pull/19609) (see
[Reproducing on Colab](#reproducing-on-colab) below).

### Reading `configs/yolo11.yaml`

```yaml
backbone:
  - [-1, 1, Timm, [2048, 'resnet101', True, True, 0, True]]  # 0: resnet101 backbone
  - [0, 1, Index, [512, 2]]     # 1: P3 - [1, 512, 80, 80]
  - [0, 1, Index, [1024, 3]]    # 2: P4 - [1, 1024, 40, 40]
  - [0, 1, Index, [2048, 4]]    # 3: P5 - [1, 2048, 20, 20]
  - [-1, 1, SPPF, [2048, 5]]    # 4: SPPF on P5
```

- **`Timm [2048, 'resnet101', True, True, 0, True]`** — instantiates a timm
  `resnet101` model as the backbone. In order, the arguments are: output
  channels (`2048`, used by Ultralytics' model parser for bookkeeping),
  the timm model name, `pretrained=True` (load ImageNet weights),
  `unwrap=True` (run in `features_only` mode so it returns intermediate
  feature maps instead of a classification logit), `truncate=0` (don't
  drop any trailing layers), and `split=True` (return the multi-scale
  features as a list so later layers can pick individual stages out of it).
- **`Index [C, i]`** — picks a single feature map out of that list by index
  `i` and tells the parser to expect `C` channels for it. A `features_only`
  ResNet101 in timm exposes 5 stages (strides 2/4/8/16/32); indices `2`, `3`,
  `4` correspond to strides 8, 16, 32 — i.e. the **P3, P4, P5** feature maps
  YOLO's neck expects, with `512`, `1024`, `2048` channels respectively.
- **`SPPF [2048, 5]`** — Spatial Pyramid Pooling – Fast, applied to the P5
  map. It pools the deepest feature map at multiple scales and concatenates
  the results, giving the network a wider receptive field before the neck
  starts up-sampling and merging P4/P3 — the same SPPF block stock YOLO
  models use, just re-attached after the ResNet101 output instead of after
  CSPDarknet.

The rest of the neck/head is stock YOLOv11: upsample-concat-C2f blocks
merging P5→P4→P3 top-down, then C2f-downsample blocks merging bottom-up
before the three-scale `Detect` head.

## Datasets

| | **OHRC** | **TMC-2** |
|---|---|---|
| Camera | Orbiter High Resolution Camera | Terrain Mapping Camera-2 |
| Ground resolution | ~0.25 m/pixel | ~5 m/pixel |
| Tile size | 304×640 px | 79×640 px |
| Annotation | Roboflow-assisted | Manual bounding boxes |
| Epochs trained | 300 | 150 |
| Roboflow project | `ch2-ohr` | `ch2-tmc` |

OHRC's much finer resolution captures small, detailed craters at the cost of
narrower ground coverage per tile; TMC-2 trades resolution for wider coverage,
so it tends to pick up larger craters across bigger areas per image. Training
the two separately keeps each model's anchors/scale expectations matched to
its own imagery instead of averaging across very different crater-to-pixel
ratios.

## Repo structure

```
.
├── notebooks/
│   ├── ohrc.ipynb       # OHRC dataset training notebook (Colab)
│   └── tmc2.ipynb       # TMC-2 dataset training notebook (Colab)
├── configs/
│   └── yolo11.yaml      # YOLOv11 model config with ResNet101/timm backbone
├── results/             # training curves + prediction samples go here (empty for now)
├── requirements.txt
└── README.md
```

## Reproducing on Colab

Both notebooks follow the same steps; only the Roboflow project and epoch
count differ.

1. **Open the notebook in Colab** (`notebooks/ohrc.ipynb` or
   `notebooks/tmc2.ipynb`) and switch the runtime to a GPU
   (`Runtime → Change runtime type → GPU`). The first cell (`!nvidia-smi`)
   confirms a GPU is attached.

2. **Set your own Roboflow API key.** The notebooks ship with the key
   replaced by `"YOUR_API_KEY"` — get your own key from your
   [Roboflow account settings](https://app.roboflow.com/) and paste it in:
   ```python
   rf = Roboflow(api_key="YOUR_API_KEY")
   ```
   Never commit a real key back into these notebooks.

3. **Check out the Ultralytics PR that adds timm backbone support**, since
   it isn't in a released version yet:
   ```bash
   git clone https://github.com/ultralytics/ultralytics
   cd ultralytics
   git fetch origin pull/19609/head:pr-19609
   git checkout pr-19609
   ```

4. **Copy `configs/yolo11.yaml` into the checked-out Ultralytics repo**, so
   it's discoverable as a model config by name:
   ```bash
   cp yolo11.yaml ultralytics/ultralytics/cfg/models/11/
   ```

5. **Load and train:**
   ```python
   from ultralytics import YOLO

   model = YOLO("yolo11.yaml")
   results = model.train(
       data="/content/<dataset>/data.yaml",
       epochs=300,   # 150 for TMC-2
       batch=8,
       imgsz=640,
   )
   ```

6. **Resuming across disconnects.** Colab sessions can drop before a
   150–300 epoch run finishes. To make that recoverable, mount Drive and
   point training output there instead of at local disk, then resume from
   the last checkpoint if the session restarts:
   ```python
   from google.colab import drive
   drive.mount("/content/drive")

   model.train(
       data="/content/<dataset>/data.yaml",
       epochs=300,
       batch=8,
       imgsz=640,
       project="/content/drive/MyDrive/lunar-crater-runs",
       resume=True,   # picks up from the last checkpoint in `project` if one exists
   )
   ```

7. **Save results.** Copy training curves (`results.png`, `confusion_matrix.png`,
   etc.) and a few prediction images from `runs/detect/train/` into this
   repo's [`results/`](results/) folder once a run finishes.

## Results

Not yet trained — the tables below are placeholders to fill in after running
the notebooks above.

### OHRC

| Metric | Result |
|---|---|
| Precision | _TBD_ |
| Recall | _TBD_ |
| mAP@0.50 | _TBD_ |
| mAP@0.50:0.95 | _TBD_ |

### TMC-2

| Metric | Result |
|---|---|
| Precision | _TBD_ |
| Recall | _TBD_ |
| mAP@0.50 | _TBD_ |
| mAP@0.50:0.95 | _TBD_ |

### Training curves & sample predictions

_Add `results.png`, `confusion_matrix.png`, and a few annotated prediction
images to [`results/`](results/) after training, and link them here._

## Credits & License

Imagery courtesy of ISRO's Chandrayaan-2 mission (OHRC and TMC-2 instruments).
Timm backbone support in Ultralytics via
[PR #19609](https://github.com/ultralytics/ultralytics/pull/19609).

This project is released under the MIT License.
