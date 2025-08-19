# Temperature Image Super-Resolution (SRResUNet)

Super-resolve low-resolution temperature (thermal) images using a lightweight **SRResUNet** model implemented in PyTorch.  
The project includes:

- A Hydra-driven **train/test** pipeline (`main.py`)
- A dataset **augmentation** pipeline (`augment.py`)
- A **PyQt GUI** app for single/multi-image inference (`ruperRes_gui.py`)
- Reproducible outputs (timestamped) and optional Weights & Biases logging

## 🧭 Repository Structure

```
├── configs/
│   └── config.yaml
├── data/
│   ├── raw/          # your raw LR/HR sources (see augment.py)
│   └── test/         # can hold ad-hoc test images
├── debug/
│   ├── debug_dataset.ipynb
│   └── debug_test.ipynb
├── outputs/          # hydra-run outputs, auto timestamped
├── src/
│   ├── data/
│   │   └── dataloader.py
│   ├── models/
│   │   └── SRResUNet.py
│   └── utils/
├── tempResEnv/       # local env folder
├── .gitignore
├── augment.py
├── main.py           # Main file for training and testing model
├── pyproject.toml
├── requirements.txt
├── setup.py
└── ruperRes_gui.py   # PyQt GUI for inference
```


## ⚙️ Environment Setup


Make and activate an environment
```bash
python -m venv <env>
source <env>/bin/activate
```

Intall the project by running the following program 
```bash
pip install -e .
```


## 📁 Data & Augmentation

### Expected final dataset layout (after augmentation)
`augment.py` builds an **augmented** dataset under `data/augmented/`:

```
data/augmented/
├── train/
│   ├── low_res/       # 120 × 160, RGB PNGs
│   └── high_res/      # 480 × 640, RGB PNGs
├── val/
│   ├── low_res/
│   └── high_res/
└── test/
    ├── low_res/
    └── high_res/
```

- LR images are 120×160; HR images are 480×640 (4× upscaling).
- Filenames **must match** across LR/HR (the dataloader pairs by name).

### Build the augmented dataset

`augment.py` expects your raw inputs in:

- `data/raw/webcam_frames7/` (LR source)
- `data/raw/flir_frames7/`   (HR source)

Run:

```bash
python augment.py
```

Key details from `augment.py`:

- Applies geometric + photometric transforms with **Albumentations**.
- Ensures HR is **480×640** and LR is **120×160**.
- Splits Train/Val/Test using the configured ratios and saves PNG pairs.


## 🧪 Train & Test (Hydra)

### Config

Edit `configs/config.yaml` (example keys):

```yaml
dataset:
  name: custom
  path: data/augmented/
model:
  name: srresunet
  in_channels: 3
  out_channels: 3
  num_filters: 32
  num_residuals: 2
  upscale_factor: 4
training:
  epochs: 100
  batch_size: 32
  learning_rate: 0.0001
other:
  wandb: false              # set to your project name to enable W&B
  run_testing: true
  run_testing_only: true    # set to false if training the model
  testing_only_model_path: outputs/2025-05-29/22-54-23/models/
```

Hydra creates a new timestamped directory under `outputs/` on each run and writes the resolved `config.yaml` there.

### Train

```bash
python main.py 
```

**What happens:**

- Datasets are loaded via `src/data/dataloader.py` (OpenCV → RGB → `[0,1]` float32, tensors shaped `(C,H,W)`).
- Model: `src/models/SRResUNet.py` with the configured channels/filters/residuals and `upscale_factor=4`.
- Training uses **L1 loss** and **Adam** optimizer.
- Best and final checkpoints saved to `outputs/<date>/<time>/models/`.

### Test (using best/final checkpoint)

**Option A — test only (no training), path given:**

```bash
python main.py \
  other.run_testing=true \
  other.run_testing_only=true \
  other.testing_only_model_path=outputs/2025-05-29/22-54-23/models/ \
  dataset.path=data/augmented
```

**Option B — after training in the same run:**

```bash
python main.py other.run_testing=true other.run_testing_only=false
```

**Outputs:**

- Per-image LR/GeneratedHR/GT saved under:
  `outputs/<date>/<time>/test_outputs/{input_lr,generated_hr,ground_truth_hr}/`
- Average test **L1 loss** printed and (if enabled) logged to W&B.


## 🖼️ GUI Inference App (PyQt)

The GUI lets you super-resolve arbitrary images (not just the dataset pairs), with **padding/cropping** logic to avoid UNet size mismatches.

Launch:

```bash
python superRes_gui.py
```

**Workflow:**

1. **Load Model (.pth)** – pick your `best_model.pth` or `final_model.pth`.
   - The app **auto-loads** the `config.yaml` next to the checkpoint if present.
2. **Open Image(s) / Directory** – select one or more images (PNG/JPG/TIF...).
3. (Optional) **Choose Save Dir** and check “Save outputs”.
4. Click **Process Current** or **Process All**.
5. Navigate with **Previous / Next** (centered buttons), or **←/→** keys.
6. The bottom bar shows a **0–100%** position slider (percentage) + per-run progress bar.

**Display:**

- Shows **Input** and **Output** previews with image sizes centered below each view.


## 📚 References & Useful Links

- PyTorch: https://pytorch.org/  
- Hydra: https://hydra.cc/  
- Albumentations: https://albumentations.ai/  
- Weights & Biases: https://wandb.ai/  
- U-Net (Ronneberger et al., 2015): https://arxiv.org/abs/1505.04597  
- Super-Resolution Residual U-Net Model  (Chen et al., 2022): https://pubs.acs.org/doi/10.1021/acsomega.2c01435

> The SRResUNet in this repo  is motivated from the above papers for super-resolution; see the original papers above for background and design intuition.

