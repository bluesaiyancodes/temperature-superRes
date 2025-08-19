#!/usr/bin/env python3
# sr_gui.py — PyQt5 GUI for Temperature Image Super-Resolution (SRResUNet)


import os
import sys
import json
import traceback
from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple

# ---- PyQt5 ----
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QSize
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QFileDialog, QLabel, QPushButton,
    QHBoxLayout, QVBoxLayout, QSlider, QTextEdit, QGroupBox, QMessageBox,
    QProgressBar, QCheckBox, QLineEdit
)
from PyQt5.QtGui import QPixmap, QImage, QKeySequence

# ---- Torch / NumPy / OpenCV / YAML ----
import numpy as np
import torch
import torch.nn.functional as F
import cv2

try:
    import yaml
except Exception:
    yaml = None

# ---- Import user's model ----
try:
    from src.models.SRResUNet import SRResUNet
except Exception as e:
    SRResUNet = None
    _IMPORT_ERROR = e

# --------------------- Defaults ---------------------

DEFAULT_CFG = {
    "model": {
        "name": "srresunet",
        "in_channels": 3,
        "out_channels": 3,
        "num_filters": 32,
        "num_residuals": 2,
        "upscale_factor": 4,
        "pretrained": False,
    },
    "other": {
        # Match main.py: full precision by default.
        "half": False,
        # UNet downsample depth (D). Dataset likely ensured multiples of 2^D already.
        "downsample_stages": 3,  # 3 -> multiple-of-8
        "tile_infer": False,
        "tile_size": 384,
        "tile_overlap": 16,
    }
}

IMG_EXT = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def is_image_file(path: str) -> bool:
    return path.lower().endswith(IMG_EXT)


# --------------------- Image utilities ---------------------

def to_qimage(arr_rgb_u8: np.ndarray) -> QImage:
    """
    arr_rgb_u8: HxWx3, dtype=uint8, RGB
    Robust against PyQt refusing memoryview: pass bytes buffer.
    """
    if arr_rgb_u8.ndim != 3 or arr_rgb_u8.shape[2] != 3 or arr_rgb_u8.dtype != np.uint8:
        raise ValueError("to_qimage expects uint8 RGB array of shape HxWx3.")
    if not arr_rgb_u8.flags["C_CONTIGUOUS"]:
        arr_rgb_u8 = np.ascontiguousarray(arr_rgb_u8)

    h, w, _ = arr_rgb_u8.shape
    bytes_per_line = 3 * w
    qimg = QImage(arr_rgb_u8.tobytes(), w, h, bytes_per_line, QImage.Format_RGB888)
    return qimg.copy()


def load_image_like_dataset(path: str, in_channels: int = 3) -> Tuple[torch.Tensor, np.ndarray]:
    """
    Replicates SuperResolutionDataset preprocessing:
      - cv2.imread(..., IMREAD_COLOR) -> BGR
      - cv2.cvtColor -> RGB
      - float32 [0,1]
      - (1,3,H,W) tensor
    Returns:
      ten  : torch.FloatTensor (1,3,H,W) in [0,1]
      disp : np.uint8 HxWx3 RGB for preview
    """
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise IOError(f"Could not read image: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)                 # RGB
    img_f32 = img.astype(np.float32) / 255.0                   # [0,1]
    if in_channels != 3:
        # Your training config uses 3; if you change it later, adapt here.
        raise ValueError(f"Model expects in_channels={in_channels}, dataset loader is 3.")
    ten = torch.from_numpy(img_f32.transpose(2, 0, 1)).unsqueeze(0)  # (1,3,H,W)
    disp = img  # uint8 RGB
    return ten, disp


def pad_to_multiple(t: torch.Tensor, mult: int) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """
    Pad (1,C,H,W) to H' and W' that are multiples of 'mult', using reflect.
    Returns (t_pad, (pad_h, pad_w)) where pad was applied at bottom/right.
    """
    h, w = t.shape[-2:]
    pad_h = (mult - h % mult) % mult
    pad_w = (mult - w % mult) % mult
    if pad_h or pad_w:
        t = F.pad(t, (0, pad_w, 0, pad_h), mode="reflect")
    return t, (pad_h, pad_w)


def crop_sr_to_original(sr: torch.Tensor, h: int, w: int, scale: int) -> torch.Tensor:
    """
    Crop SR (1,C,H',W') back to (1,C,h*scale,w*scale).
    """
    return sr[..., :h * scale, :w * scale]


# --------------------- Worker thread ---------------------

class SRWorker(QThread):
    progress = pyqtSignal(int)                 # 0..100
    message = pyqtSignal(str)                  # log text
    result_one = pyqtSignal(int, QImage)       # (index, qimage)
    finished_all = pyqtSignal()

    def __init__(self, model, device, image_paths: List[str], indices: List[int],
                 out_dir: Optional[str], cfg: Dict, parent=None):
        super().__init__(parent)
        self.model = model
        self.device = device
        self.image_paths = image_paths
        self.indices = indices
        self.out_dir = out_dir
        self.cfg = cfg
        self._abort = False

    def abort(self):
        self._abort = True

    def run(self):
        try:
            self.model.eval()
            use_half = bool(self.cfg.get("other", {}).get("half", False)) and self.device.type == "cuda"
            upscale = int(self.cfg["model"].get("upscale_factor", 4))
            in_ch = int(self.cfg["model"].get("in_channels", 3))
            D = int(self.cfg["other"].get("downsample_stages", 3))
            mult = 1 << D

            for k, idx in enumerate(self.indices):
                if self._abort:
                    break
                path = self.image_paths[idx]
                self.message.emit(f"Processing [{idx + 1}/{len(self.image_paths)}]: {os.path.basename(path)}")

                # 1) Load like dataset
                ten, disp_rgb = load_image_like_dataset(path, in_channels=in_ch)  # (1,3,H,W), float32 [0,1]
                self.message.emit(f"Input tensor shape before pad: {tuple(ten.shape)}")
                ten = ten.to(self.device)

                # 2) Pad to multiple of 2^D (dataset-friendly), then forward
                h, w = ten.shape[-2:]
                ten_pad, _ = pad_to_multiple(ten, mult)

                with torch.no_grad():
                    if use_half:
                        self.model.half()
                        ten_pad = ten_pad.half()
                    sr = self.model(ten_pad)  # (1,C,H',W') expected in [0,1]
                    sr = sr.float()

                # 3) Crop back to original SR size
                sr = crop_sr_to_original(sr, h, w, upscale)

                # 4) To uint8 RGB for display/save
                sr_np = sr.squeeze(0).cpu().numpy()  # CxHxW
                sr_np = np.transpose(sr_np, (1, 2, 0))  # HxWxC
                sr_u8 = (sr_np * 255.0 + 0.5).clip(0, 255).astype(np.uint8)
                qimg = to_qimage(sr_u8[:, :, :3])

                # 5) Save if requested
                if self.out_dir:
                    os.makedirs(self.out_dir, exist_ok=True)
                    base = os.path.splitext(os.path.basename(path))[0]
                    out_path = os.path.join(self.out_dir, f"{base}_SRx{upscale}.png")
                    # Save RGB PNG
                    cv2.imwrite(out_path, cv2.cvtColor(sr_u8[:, :, :3], cv2.COLOR_RGB2BGR))
                    self.message.emit(f"Saved: {out_path}")

                # 6) Emit to UI
                self.result_one.emit(idx, qimg)
                self.progress.emit(int((k + 1) / len(self.indices) * 100))

        except Exception:
            self.message.emit("ERROR:\n" + "".join(traceback.format_exc()))
        finally:
            self.finished_all.emit()


# --------------------- GUI main window ---------------------

@dataclass
class Session:
    image_paths: List[str]
    outputs: Dict[int, QImage]
    current_index: int
    model_loaded: bool
    device: torch.device
    cfg: Dict


class SRWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Temperature Image Super-Resolution")
        self.resize(1280, 820)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.sess = Session(
            image_paths=[],
            outputs={},
            current_index=0,
            model_loaded=False,
            device=device,
            cfg=json.loads(json.dumps(DEFAULT_CFG)),  # deep copy
        )
        self.model = None
        self.worker: Optional[SRWorker] = None
        self.out_dir: Optional[str] = None
        self._updating_slider_programmatically = False  # guard to avoid loops

        self._build_ui()
        self._log(f"Device: {self.sess.device}")
        if SRResUNet is None:
            self._log("Could not import SRResUNet from src.models.SRResUNet.")
            self._log(f"Import error:\n{_IMPORT_ERROR}")

    # ---- UI layout ----
    def _build_ui(self):
        root = QWidget()
        self.setCentralWidget(root)

        # Top-left: file controls
        self.btn_load_img = QPushButton("Open Image(s)")
        self.btn_load_dir = QPushButton("Open Directory")
        self.btn_load_model = QPushButton("Load Model (.pth)")
        self.cfg_line = QLineEdit("auto (use config.yaml near checkpoint or defaults)")
        self.btn_choose_cfg = QPushButton("Pick Config YAML")

        # Inference controls
        self.btn_process_current = QPushButton("Process Current")
        self.btn_process_all = QPushButton("Process All")
        self.chk_save = QCheckBox("Save outputs")
        self.chk_save.setChecked(True)
        self.btn_choose_out = QPushButton("Choose Save Dir")
        self.out_dir_line = QLineEdit("")
        self.out_dir_line.setPlaceholderText("<no directory selected>")
        self.out_dir_line.setReadOnly(True)

        # Assemble top bar
        top = QHBoxLayout()
        top.addWidget(self.btn_load_img)
        top.addWidget(self.btn_load_dir)
        top.addSpacing(8)
        top.addWidget(self.btn_load_model)
        top.addWidget(self.cfg_line, 1)
        top.addWidget(self.btn_choose_cfg)
        top.addSpacing(8)
        top.addWidget(self.btn_process_current)
        top.addWidget(self.btn_process_all)

        save_row = QHBoxLayout()
        save_row.addWidget(self.chk_save)
        save_row.addWidget(self.btn_choose_out)
        save_row.addWidget(self.out_dir_line, 1)

        # Image previews + size labels
        self.lbl_in = QLabel("Input")
        self.lbl_in.setAlignment(Qt.AlignCenter)
        self.lbl_in.setMinimumSize(QSize(360, 360))
        self.lbl_in.setStyleSheet("border: 1px solid #666; background: #111; color: #ccc;")

        self.lbl_in_size = QLabel("—")
        self.lbl_in_size.setAlignment(Qt.AlignCenter)
        self.lbl_in_size.setStyleSheet("color: #aaa; padding: 4px;")

        col_in = QVBoxLayout()
        col_in.addWidget(self.lbl_in, 1)
        col_in.addWidget(self.lbl_in_size, 0, alignment=Qt.AlignCenter)

        self.lbl_out = QLabel("Output")
        self.lbl_out.setAlignment(Qt.AlignCenter)
        self.lbl_out.setMinimumSize(QSize(360, 360))
        self.lbl_out.setStyleSheet("border: 1px solid #666; background: #111; color: #ccc;")

        self.lbl_out_size = QLabel("—")
        self.lbl_out_size.setAlignment(Qt.AlignCenter)
        self.lbl_out_size.setStyleSheet("color: #aaa; padding: 4px;")

        col_out = QVBoxLayout()
        col_out.addWidget(self.lbl_out, 1)
        col_out.addWidget(self.lbl_out_size, 0, alignment=Qt.AlignCenter)

        img_row = QHBoxLayout()
        img_row.addLayout(col_in, 1)
        img_row.addLayout(col_out, 1)

        # --- Centered navigation buttons (middle, center-aligned) ---
        self.btn_prev = QPushButton("◀ Previous")
        self.btn_next = QPushButton("Next ▶")
        nav_row = QHBoxLayout()
        nav_row.addStretch(1)
        nav_row.addWidget(self.btn_prev)
        nav_row.addSpacing(12)
        nav_row.addWidget(self.btn_next)
        nav_row.addStretch(1)

        # Percentage "scroll bar" (0..100) + percent label + status + inference progress
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, 100)
        self.slider.setSingleStep(1)
        self.slider.setPageStep(5)
        self.slider.setTracking(True)

        self.lbl_percent = QLabel("0%")
        self.lbl_percent.setAlignment(Qt.AlignCenter)
        self.lbl_status = QLabel("0 / 0")

        self.prog = QProgressBar()
        self.prog.setValue(0)

        slider_row = QHBoxLayout()
        slider_row.addWidget(QLabel("0%"))
        slider_row.addWidget(self.slider, 1)
        slider_row.addWidget(QLabel("100%"))
        slider_row.addSpacing(10)
        slider_row.addWidget(self.lbl_percent)
        slider_row.addSpacing(10)
        slider_row.addWidget(self.lbl_status)
        slider_row.addSpacing(10)
        slider_row.addWidget(self.prog)

        # Log
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setStyleSheet("border: 1px solid #666;")

        grp_top = QGroupBox("Select Data / Model")
        g1 = QVBoxLayout()
        g1.addLayout(top)
        g1.addLayout(save_row)
        grp_top.setLayout(g1)

        grp_views = QGroupBox("Preview")
        g2 = QVBoxLayout()
        g2.addLayout(img_row)
        grp_views.setLayout(g2)

        grp_log = QGroupBox("Log")
        g3 = QVBoxLayout()
        g3.addWidget(self.log)
        grp_log.setLayout(g3)

        layout = QVBoxLayout()
        layout.addWidget(grp_top)
        layout.addWidget(grp_views, 1)
        layout.addLayout(nav_row)       # centered prev/next row
        layout.addLayout(slider_row)    # percentage bar row
        layout.addWidget(grp_log, 1)
        root.setLayout(layout)

        # Signals
        self.btn_load_img.clicked.connect(self._choose_images)
        self.btn_load_dir.clicked.connect(self._choose_dir)
        self.btn_load_model.clicked.connect(self._choose_model)
        self.btn_choose_cfg.clicked.connect(self._choose_cfg)
        self.btn_choose_out.clicked.connect(self._choose_out_dir)
        self.btn_process_current.clicked.connect(lambda: self._start_infer(all_items=False))
        self.btn_process_all.clicked.connect(lambda: self._start_infer(all_items=True))

        self.btn_prev.clicked.connect(self._go_prev)
        self.btn_next.clicked.connect(self._go_next)
        self.slider.valueChanged.connect(self._on_percent_slide)

        # Keyboard navigation (Left/Right)
        self.btn_prev.setShortcut(QKeySequence(Qt.Key_Left))
        self.btn_next.setShortcut(QKeySequence(Qt.Key_Right))

    # ---- Logging ----
    def _log(self, text: str):
        self.log.append(text)
        self.log.ensureCursorVisible()

    # ---- Index/percent mapping ----
    @staticmethod
    def _idx_to_percent(idx: int, n: int) -> int:
        if n <= 1:
            return 0
        return int(round(100 * idx / (n - 1)))

    @staticmethod
    def _percent_to_idx(p: int, n: int) -> int:
        if n <= 1:
            return 0
        p = max(0, min(100, int(p)))
        return int(round(p * (n - 1) / 100))

    def _sync_slider_to_index(self):
        n = len(self.sess.image_paths)
        p = self._idx_to_percent(self.sess.current_index, n)
        self._updating_slider_programmatically = True
        try:
            self.slider.setValue(p)
            self.lbl_percent.setText(f"{p}%")
        finally:
            self._updating_slider_programmatically = False

    # ---- File selection ----
    def _choose_images(self):
        files, _ = QFileDialog.getOpenFileNames(self, "Select image(s)", "", "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)")
        if not files:
            return
        files = [f for f in files if is_image_file(f)]
        self._set_images(files)

    def _choose_dir(self):
        d = QFileDialog.getExistingDirectory(self, "Select directory with images")
        if not d:
            return
        files = [os.path.join(d, f) for f in sorted(os.listdir(d)) if is_image_file(os.path.join(d, f))]
        if not files:
            QMessageBox.information(self, "No images", "This folder has no supported images.")
            return
        self._set_images(files)

    def _set_images(self, files: List[str]):
        self.sess.image_paths = files
        self.sess.outputs.clear()
        self.sess.current_index = 0
        self._update_status()
        self._update_views()
        self._sync_slider_to_index()
        self._log(f"Loaded {len(files)} image(s).")

    # ---- Output dir ----
    def _choose_out_dir(self):
        d = QFileDialog.getExistingDirectory(self, "Choose output directory")
        if d:
            self.out_dir = d
            # Show it somewhere useful
            self._log(f"Output directory: {d}")

    # ---- Config & Model ----
    def _choose_cfg(self):
        path, _ = QFileDialog.getOpenFileName(self, "Pick config.yaml (optional)", "", "YAML (*.yaml *.yml)")
        if not path:
            return
        if yaml is None:
            QMessageBox.warning(self, "pyyaml missing", "pyyaml not installed. `pip install pyyaml`")
            return
        try:
            with open(path, "r") as f:
                cfg = yaml.safe_load(f)
            self._merge_cfg(cfg)
            self.cfg_line.setText(path)
            self._log(f"Loaded config from: {path}")
        except Exception as e:
            QMessageBox.critical(self, "Config error", str(e))

    def _merge_cfg(self, new_cfg: Dict):
        for k in ("model", "other"):
            if k in new_cfg and isinstance(new_cfg[k], dict):
                self.sess.cfg.setdefault(k, {})
                self.sess.cfg[k].update(new_cfg[k])

    def _choose_model(self):
        ckpt, _ = QFileDialog.getOpenFileName(self, "Select checkpoint (.pth)", "", "PyTorch Checkpoint (*.pth *.pt)")
        if not ckpt:
            return
        if SRResUNet is None:
            QMessageBox.critical(
                self, "Model import failed",
                "Could not import SRResUNet from src.models.SRResUNet.\n"
                "Ensure your PYTHONPATH includes the project root.\n\n"
                f"Import error:\n{_IMPORT_ERROR}"
            )
            return

        # Auto-load config.yaml next to checkpoint if user hasn't picked one
        if yaml is not None and self.cfg_line.text().strip().startswith("auto"):
            try:
                cand = os.path.join(os.path.dirname(ckpt), "config.yaml")
                if not os.path.exists(cand):
                    cand = os.path.join(os.path.dirname(os.path.dirname(ckpt)), "config.yaml")
                if os.path.exists(cand):
                    with open(cand, "r") as f:
                        cfg = yaml.safe_load(f)
                    self._merge_cfg(cfg)
                    self.cfg_line.setText(cand)
                    self._log(f"Auto-loaded config: {cand}")
            except Exception as e:
                self._log(f"Auto-load config failed: {e}")

        try:
            mcfg = self.sess.cfg["model"]
            self.model = SRResUNet(
                in_channels=int(mcfg.get("in_channels", 3)),
                out_channels=int(mcfg.get("out_channels", 3)),
                num_filters=int(mcfg.get("num_filters", 32)),
                num_residuals=int(mcfg.get("num_residuals", 2)),
                upscale_factor=int(mcfg.get("upscale_factor", 4)),
            ).to(self.sess.device)

            ck = torch.load(ckpt, map_location=self.sess.device)
            state = ck["model_state_dict"] if isinstance(ck, dict) and "model_state_dict" in ck else ck
            self.model.load_state_dict(state, strict=True)
            self.model.eval()
            self.sess.model_loaded = True
            self._log(f"Loaded model: {ckpt}")
        except Exception as e:
            self.sess.model_loaded = False
            self.model = None
            QMessageBox.critical(self, "Load failed", f"Could not load checkpoint:\n{e}\n\nTrace:\n{traceback.format_exc()}")

    # ---- Slider & Views ----
    def _on_percent_slide(self, val: int):
        # map 0..100 to index 0..N-1
        if self._updating_slider_programmatically:
            return
        n = len(self.sess.image_paths)
        idx = self._percent_to_idx(val, n)
        self.lbl_percent.setText(f"{val}%")
        if idx != self.sess.current_index:
            self.sess.current_index = idx
            self._update_status()
            self._update_views()

    def _update_status(self):
        n = len(self.sess.image_paths)
        i = (self.sess.current_index + 1) if n > 0 else 0
        self.lbl_status.setText(f"{i} / {n}")

    def _update_views(self):
        if not self.sess.image_paths:
            self.lbl_in.setText("Input")
            self.lbl_out.setText("Output")
            self.lbl_in_size.setText("—")
            self.lbl_out_size.setText("—")
            return

        path = self.sess.image_paths[self.sess.current_index]
        try:
            # Load like dataset for preview
            _, disp_rgb = load_image_like_dataset(path, in_channels=int(self.sess.cfg["model"]["in_channels"]))
            qim = to_qimage(disp_rgb)
            self._set_label_image(self.lbl_in, qim)
            h, w, _ = disp_rgb.shape
            self.lbl_in_size.setText(f"{w} × {h} px")
        except Exception as e:
            self.lbl_in.setText(f"Failed to load:\n{os.path.basename(path)}")
            self.lbl_in_size.setText("—")
            self._log(f"Preview error: {e}")

        q = self.sess.outputs.get(self.sess.current_index, None)
        if q is not None:
            self._set_label_image(self.lbl_out, q)
            self.lbl_out_size.setText(f"{q.width()} × {q.height()} px")
        else:
            self.lbl_out.setText("Output")
            self.lbl_out_size.setText("—")

        # keep slider percentage in sync with index (when update came from buttons or keys)
        self._sync_slider_to_index()

    def _set_label_image(self, lbl: QLabel, qimg: QImage):
        pix = QPixmap.fromImage(qimg)
        pix = pix.scaled(lbl.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        lbl.setPixmap(pix)

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._update_views()

    # ---- Navigation: prev/next & keyboard ----
    def _go_prev(self):
        if self.sess.current_index > 0:
            self.sess.current_index -= 1
            self._update_status()
            self._update_views()

    def _go_next(self):
        n = len(self.sess.image_paths)
        if self.sess.current_index < max(0, n - 1):
            self.sess.current_index += 1
            self._update_status()
            self._update_views()

    # ---- Inference ----
    def _start_infer(self, all_items: bool):
        if not self.sess.model_loaded or self.model is None:
            QMessageBox.information(self, "No model", "Please load a model checkpoint first.")
            return
        if not self.sess.image_paths:
            QMessageBox.information(self, "No images", "Please open image(s) or a directory first.")
            return
        if self.chk_save.isChecked() and not self.out_dir:
            self._choose_out_dir()

        if self.worker is not None and self.worker.isRunning():
            QMessageBox.information(self, "Busy", "Inference already running.")
            return

        indices = list(range(len(self.sess.image_paths))) if all_items else [self.sess.current_index]
        out_dir = self.out_dir if self.chk_save.isChecked() else None

        self.worker = SRWorker(self.model, self.sess.device, self.sess.image_paths, indices, out_dir, self.sess.cfg)
        self.worker.progress.connect(self.prog.setValue)
        self.worker.message.connect(self._log)
        self.worker.result_one.connect(self._on_result_one)
        self.worker.finished_all.connect(self._on_finished)
        self.prog.setValue(0)
        self._log(f"Starting inference on {len(indices)} image(s)…")
        self.worker.start()

    def _on_result_one(self, idx: int, qimg: QImage):
        self.sess.outputs[idx] = qimg
        if idx == self.sess.current_index:
            self._set_label_image(self.lbl_out, qimg)
            self.lbl_out_size.setText(f"{qimg.width()} × {qimg.height()} px")

    def _on_finished(self):
        self._log("Done.")
        self.worker = None


def main():
    app = QApplication(sys.argv)
    w = SRWindow()
    w.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
