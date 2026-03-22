# ---------------------------------------------------------------------------
# gui.py  — deferred heavy imports
#
# PySide6 MUST stay at top level (needed to define QWidget subclasses).
# PIL, numpy, processing, emissive, and _ipc are all imported lazily.
# ---------------------------------------------------------------------------
from __future__ import annotations

from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QHBoxLayout,
    QFileDialog,
    QSlider,
    QSpinBox,
    QDoubleSpinBox,
    QCheckBox,
    QGroupBox,
    QSizePolicy,
    QComboBox,
    QToolBar,
    QToolButton,
    QMessageBox,
    QDialog,
    QDialogButtonBox,
    QTabWidget,
    QScrollArea,
    QColorDialog,
    QFrame,
)
from PySide6.QtCore import Qt, QTimer, Signal, QSize, QPointF, QRectF, QPropertyAnimation, QEasingCurve
from PySide6.QtGui import QPixmap, QAction, QWheelEvent, QPainter, QCursor, QPalette, QColor
import sys
import json
import pathlib
import traceback
import concurrent.futures
import multiprocessing
import threading

# ---------------------------------------------------------------------------
# Preset discovery — scan the ``presets/`` folder next to this file for any
# *.json files.  Each file is one preset; ``_name`` inside the JSON is the
# display label (falls back to the filename stem).
# ---------------------------------------------------------------------------

_PRESETS_DIR = pathlib.Path(__file__).parent / "presets"


def _load_emissive_presets() -> list[dict]:
    """Return a list of preset dicts loaded from presets/*.json.

    Each dict contains all the raw JSON values plus a synthetic ``_path`` key
    with the source file.  JSON arrays that represent RGB triples are converted
    to tuples so the rest of the code doesn't need to care about the difference.
    """
    presets = []
    if not _PRESETS_DIR.exists():
        return presets
    for p in sorted(_PRESETS_DIR.glob("*.json")):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            # Coerce JSON arrays that represent RGB triples to tuples
            for key in ("gm_shadow", "gm_mid", "gm_highlight", "bloom_color"):
                if key in data and isinstance(data[key], list):
                    data[key] = tuple(data[key])
            data.setdefault("_name", p.stem.replace("_", " ").title())
            data["_path"] = str(p)
            presets.append(data)
        except Exception as exc:
            print(f"[presets] failed to load {p}: {exc}")
    return presets


# ---------------------------------------------------------------------------
# Lazy-import accessors — cached singletons, safe under the GIL
# ---------------------------------------------------------------------------

_processing_mod = None
_emissive_mod   = None
_ipc_mod        = None
_PIL_Image      = None
_PIL_ImageQt    = None
_PIL_ImageFilter = None


def _get_processing():
    global _processing_mod
    if _processing_mod is None:
        from . import processing as _m
        _processing_mod = _m
    return _processing_mod


def _get_emissive():
    global _emissive_mod
    if _emissive_mod is None:
        from . import emissive as _m
        _emissive_mod = _m
    return _emissive_mod


def _get_ipc():
    global _ipc_mod
    if _ipc_mod is None:
        from . import _ipc as _m
        _ipc_mod = _m
    return _ipc_mod


def _get_PIL_Image():
    global _PIL_Image
    if _PIL_Image is None:
        from PIL import Image as _m
        _PIL_Image = _m
    return _PIL_Image


def _get_PIL_ImageQt():
    global _PIL_ImageQt
    if _PIL_ImageQt is None:
        from PIL.ImageQt import ImageQt as _m
        _PIL_ImageQt = _m
    return _PIL_ImageQt


def _get_PIL_ImageFilter():
    global _PIL_ImageFilter
    if _PIL_ImageFilter is None:
        from PIL import ImageFilter as _m
        _PIL_ImageFilter = _m
    return _PIL_ImageFilter


# ---------------------------------------------------------------------------
# Canvas widget — handles all pan / zoom rendering internally
# ---------------------------------------------------------------------------

class _PreviewCanvas(QWidget):
    """A widget that draws a QPixmap with pan and zoom."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.setCursor(Qt.OpenHandCursor)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumSize(200, 200)
        self.setStyleSheet("background-color: #222;")

        self._pixmap: QPixmap | None = None
        self._ref_size: tuple[int, int] | None = None
        self._zoom: float = 0.0
        self._pan_offset = QPointF(0, 0)

        self._drag_active = False
        self._drag_start_mouse = QPointF()
        self._drag_start_offset = QPointF()

    def set_pixmap(self, pix: QPixmap | None, ref_size: tuple[int, int] | None = None):
        self._pixmap = pix
        if ref_size is not None:
            self._ref_size = ref_size
        elif pix is not None:
            self._ref_size = (pix.width(), pix.height())
        else:
            self._ref_size = None
        self._zoom = 0.0
        self._pan_offset = QPointF(0, 0)
        self.update()

    def set_pixmap_keep_view(self, pix: QPixmap | None):
        self._pixmap = pix
        self.update()

    def reset_view(self):
        self._zoom = 0.0
        self._pan_offset = QPointF(0, 0)
        self.update()

    def apply_zoom(self, factor: float, anchor: QPointF | None = None):
        if self._pixmap is None or self._ref_size is None:
            return
        old_zoom = self._effective_zoom()
        new_zoom = max(0.05, min(32.0, old_zoom * factor))
        if abs(new_zoom - old_zoom) < 1e-9:
            return
        if anchor is None:
            anchor = QPointF(self.width() / 2, self.height() / 2)
        if self._zoom <= 0:
            rw, rh = self._ref_size
            sw = rw * old_zoom
            sh = rh * old_zoom
            self._pan_offset = QPointF(
                (self.width() - sw) / 2,
                (self.height() - sh) / 2,
            )
        img_x = (anchor.x() - self._pan_offset.x()) / old_zoom
        img_y = (anchor.y() - self._pan_offset.y()) / old_zoom
        self._zoom = new_zoom
        self._pan_offset = QPointF(
            anchor.x() - img_x * new_zoom,
            anchor.y() - img_y * new_zoom,
        )
        self._clamp_pan()
        self.update()
        return new_zoom

    def current_zoom(self) -> float:
        return self._effective_zoom()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.SmoothPixmapTransform)
        bg = self.palette().color(QPalette.Window)
        painter.fillRect(self.rect(), bg)
        if self._pixmap is None or self._ref_size is None:
            painter.setPen(self.palette().color(QPalette.WindowText))
            painter.drawText(self.rect(), Qt.AlignCenter, "No image loaded")
            return
        rw, rh = self._ref_size
        z = self._effective_zoom()
        sw = rw * z
        sh = rh * z
        if self._zoom <= 0:
            ox = (self.width() - sw) / 2
            oy = (self.height() - sh) / 2
        else:
            ox = self._pan_offset.x()
            oy = self._pan_offset.y()
        painter.drawPixmap(QRectF(ox, oy, sw, sh), self._pixmap,
                           QRectF(0, 0, self._pixmap.width(), self._pixmap.height()))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._zoom <= 0:
            self._pan_offset = self._fit_offset(self._pixmap)
        else:
            self._clamp_pan()
        self.update()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and self._pixmap is not None:
            self._drag_active = True
            self._drag_start_mouse = QPointF(event.position())
            if self._zoom <= 0 and self._ref_size is not None:
                rw, rh = self._ref_size
                z = self._effective_zoom()
                self._zoom = z
                self._pan_offset = QPointF(
                    (self.width() - rw * z) / 2,
                    (self.height() - rh * z) / 2,
                )
            self._drag_start_offset = QPointF(self._pan_offset)
            self.setCursor(Qt.ClosedHandCursor)

    def mouseMoveEvent(self, event):
        if self._drag_active:
            delta = QPointF(event.position()) - self._drag_start_mouse
            self._pan_offset = self._drag_start_offset + delta
            self._clamp_pan()
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._drag_active = False
            self.setCursor(Qt.OpenHandCursor)

    def _effective_zoom(self) -> float:
        rw, rh = self._ref_size if self._ref_size else (
            (self._pixmap.width(), self._pixmap.height()) if self._pixmap else (1, 1)
        )
        if self._zoom > 0:
            return self._zoom
        ww, wh = self.width(), self.height()
        if rw == 0 or rh == 0:
            return 1.0
        return min(ww / rw, wh / rh)

    def _fit_offset(self, pix: QPixmap | None) -> QPointF:
        return QPointF(0, 0)

    def _clamp_pan(self):
        if self._ref_size is None:
            return
        rw, rh = self._ref_size
        z = self._effective_zoom()
        sw = rw * z
        sh = rh * z
        ww, wh = self.width(), self.height()
        margin_x = min(sw, ww) * 0.5
        margin_y = min(sh, wh) * 0.5
        ox = self._pan_offset.x()
        oy = self._pan_offset.y()
        ox = max(-sw + margin_x, min(ww - margin_x, ox))
        oy = max(-sh + margin_y, min(wh - margin_y, oy))
        self._pan_offset = QPointF(ox, oy)


# ---------------------------------------------------------------------------
# Collapsible group box
# ---------------------------------------------------------------------------

class CollapsibleGroupBox(QWidget):
    def __init__(self, title: str = "", parent=None, collapsed: bool = False):
        super().__init__(parent)
        self._collapsed = collapsed

        self._toggle_btn = QToolButton()
        self._toggle_btn.setStyleSheet("QToolButton { border: none; font-weight: bold; }")
        self._toggle_btn.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self._toggle_btn.setText(title)
        self._toggle_btn.setArrowType(Qt.DownArrow if not collapsed else Qt.RightArrow)
        self._toggle_btn.setCheckable(True)
        self._toggle_btn.setChecked(not collapsed)
        self._toggle_btn.toggled.connect(self._on_toggle)

        self._content = QWidget()
        self._content_layout = QVBoxLayout()
        self._content_layout.setContentsMargins(4, 0, 0, 0)
        self._content.setLayout(self._content_layout)
        self._content.setVisible(not collapsed)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self._toggle_btn)
        layout.addWidget(self._content)

    def content_layout(self) -> QVBoxLayout:
        return self._content_layout

    def addWidget(self, widget):
        self._content_layout.addWidget(widget)

    def _on_toggle(self, checked: bool):
        self._collapsed = not checked
        self._toggle_btn.setArrowType(Qt.DownArrow if checked else Qt.RightArrow)
        self._content.setVisible(checked)


# ---------------------------------------------------------------------------
# Color picker button
# ---------------------------------------------------------------------------

class ColorButton(QPushButton):
    color_changed = Signal(object)

    def __init__(self, color: tuple[int, int, int] = (0, 0, 0), parent=None):
        super().__init__(parent)
        self._color = QColor(*color)
        self._update_style()
        self.setFixedHeight(24)
        self.clicked.connect(self._pick)

    def color(self) -> tuple[int, int, int]:
        return (self._color.red(), self._color.green(), self._color.blue())

    def set_color(self, rgb: tuple[int, int, int]):
        self._color = QColor(*rgb)
        self._update_style()

    def _update_style(self):
        r, g, b = self._color.red(), self._color.green(), self._color.blue()
        text_col = "#fff" if (r * 0.299 + g * 0.587 + b * 0.114) < 128 else "#000"
        self.setStyleSheet(
            f"background-color: rgb({r},{g},{b}); color: {text_col}; border: 1px solid #555;"
        )
        self.setText(f"({r}, {g}, {b})")

    def _pick(self):
        c = QColorDialog.getColor(self._color, self, "Choose colour")
        if c.isValid():
            self._color = c
            self._update_style()
            self.color_changed.emit(self.color())


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------

class MainWindow(QMainWindow):
    processing_done = Signal(object)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Python prototype")
        self.image = None
        self._source_image = None   # original load — never mutated; all jobs read from here
        self._working_image = None
        self._preview_image = None
        self._undo_stack = []
        self._redo_stack = []
        self._showing_original = False

        # ProcessPoolExecutor — worker process starts lazily on first submit
        self._process_pool = concurrent.futures.ProcessPoolExecutor(
            max_workers=1,
            mp_context=multiprocessing.get_context("spawn"),
        )
        self._current_future = None
        self._pending_params = None
        self._current_job_commit: bool = False
        self.processing_done.connect(self._on_future_done)

        # Pre-warm the worker process in the background so the first real job
        # doesn't pay the subprocess-spawn penalty.  Uses a daemon thread so it
        # doesn't block app shutdown.
        threading.Thread(target=self._prewarm_worker, daemon=True).start()

        # ── Menu bar ──────────────────────────────────────────────────────────
        menubar = self.menuBar()

        file_menu = menubar.addMenu("File")
        open_action = QAction("Open…", self, shortcut="Ctrl+O", triggered=self.open_image)
        save_action = QAction("Save…", self, shortcut="Ctrl+S", triggered=self.save_result)
        file_menu.addAction(open_action)
        file_menu.addAction(save_action)
        file_menu.addSeparator()
        file_menu.addAction(QAction("Exit", self, triggered=self.close))

        edit_menu = menubar.addMenu("Edit")
        self._undo_action = QAction("Undo", self, shortcut="Ctrl+Z", triggered=self._undo)
        self._undo_action.setEnabled(False)
        self._redo_action = QAction("Redo", self, shortcut="Ctrl+Y", triggered=self._redo)
        self._redo_action.setEnabled(False)
        edit_menu.addAction(self._undo_action)
        edit_menu.addAction(self._redo_action)
        edit_menu.addSeparator()
        edit_menu.addAction(QAction("Preferences…", self))

        view_menu = menubar.addMenu("View")
        view_menu.addAction(QAction("Zoom In\tCtrl++", self))
        view_menu.addAction(QAction("Zoom Out\tCtrl+-", self))
        view_menu.addAction(QAction("Fit to Window", self, triggered=self._zoom_fit))
        view_menu.addSeparator()
        view_menu.addAction(QAction("Reset Layout", self))

        help_menu = menubar.addMenu("Help")
        help_menu.addAction(QAction("Documentation", self))
        help_menu.addSeparator()
        help_menu.addAction(QAction("About…", self, triggered=self._show_about))
        help_menu.addAction(QAction("Settings…", self, triggered=self._show_settings))

        # ── Central layout ────────────────────────────────────────────────────
        central = QWidget()
        central_layout = QHBoxLayout()
        central.setLayout(central_layout)
        self.setCentralWidget(central)

        left_group = QGroupBox("Actions")
        left_layout = QVBoxLayout()
        left_group.setLayout(left_layout)
        self.open_btn = QPushButton("Open Image")
        self.save_btn = QPushButton("Save Result")
        left_layout.addWidget(self.open_btn)
        left_layout.addWidget(self.save_btn)
        left_layout.addStretch()

        self._right_tabs = QTabWidget()

        # ── Quantization tab ─────────────────────────────────────────────────
        quant_widget = QWidget()
        right_layout = QVBoxLayout()
        quant_widget.setLayout(right_layout)

        self.scale_slider = QSlider(Qt.Horizontal)
        self.scale_slider.setRange(1, 400)
        self.scale_slider.setValue(100)
        self.scale_spin = QSpinBox()
        self.scale_spin.setRange(1, 400)
        self.scale_spin.setValue(100)
        self.scale_spin.setSuffix(" %")

        self.blur_slider = QSlider(Qt.Horizontal)
        self.blur_slider.setRange(0, 50)
        self.blur_slider.setValue(0)
        self.blur_spin = QDoubleSpinBox()
        self.blur_spin.setRange(0.0, 25.0)
        self.blur_spin.setSingleStep(0.5)
        self.blur_spin.setDecimals(1)
        self.blur_spin.setValue(0.0)

        self.colors_spin = QSpinBox()
        self.colors_spin.setRange(2, 256)
        self.colors_spin.setValue(256)

        self.dither_check = QCheckBox("Dither")
        self.dither_check.setChecked(False)

        self.dither_method_combo = QComboBox()
        self.dither_method_combo.addItems(["Bayer (k-means)", "Floyd–Steinberg"])

        self.resample_combo = QComboBox()
        self.resample_combo.addItems(["Nearest", "Bilinear", "Bicubic", "Lanczos"])
        self.resample_map = {
            0: "NEAREST", 1: "BILINEAR", 2: "BICUBIC", 3: "LANCZOS",
        }

        right_layout.addWidget(QLabel("Scale"))
        sc = QWidget(); sl = QHBoxLayout(); sl.setContentsMargins(0,0,0,0); sc.setLayout(sl)
        sl.addWidget(self.scale_slider); sl.addWidget(self.scale_spin)
        right_layout.addWidget(sc)

        right_layout.addWidget(QLabel("Blur"))
        bc = QWidget(); bl = QHBoxLayout(); bl.setContentsMargins(0,0,0,0); bc.setLayout(bl)
        bl.addWidget(self.blur_slider); bl.addWidget(self.blur_spin)
        right_layout.addWidget(bc)

        right_layout.addWidget(QLabel("Colors"))
        right_layout.addWidget(self.colors_spin)
        right_layout.addWidget(self.dither_check)
        right_layout.addWidget(self.dither_method_combo)
        right_layout.addWidget(QLabel("Resample"))
        right_layout.addWidget(self.resample_combo)
        right_layout.addStretch()

        quant_reset_btn = QPushButton("\u21ba  Reset")
        quant_reset_btn.setToolTip("Reset all quantization parameters to their defaults")
        quant_reset_btn.clicked.connect(self._reset_quant_params)
        right_layout.addWidget(quant_reset_btn)

        self._right_tabs.addTab(quant_widget, "Quantization")

        self._preview_timer = QTimer(self)
        self._preview_timer.setSingleShot(True)
        self._preview_timer.setInterval(300)
        self._preview_timer.timeout.connect(self._do_preview)

        self._build_emissive_tab()
        self._right_tabs.currentChanged.connect(self._on_tab_changed)

        self.preview = _PreviewCanvas()
        self.preview.installEventFilter(self)

        central_layout.addWidget(left_group)
        central_layout.addWidget(self.preview, 1)
        central_layout.addWidget(self._right_tabs)

        _PANEL_WIDTH = 250
        left_group.setFixedWidth(_PANEL_WIDTH)
        self._right_tabs.setFixedWidth(_PANEL_WIDTH)

        # ── Bottom toolbar ────────────────────────────────────────────────────
        bottom_toolbar = QToolBar("Tools", self)
        bottom_toolbar.setMovable(False)
        bottom_toolbar.setFloatable(False)
        bottom_toolbar.setIconSize(QSize(16, 16))
        self.addToolBar(Qt.BottomToolBarArea, bottom_toolbar)

        self._before_btn = QToolButton()
        self._before_btn.setText("⬛ Before / After")
        self._before_btn.setToolTip("Hold to compare against the original file")
        self._before_btn.pressed.connect(self._show_before)
        self._before_btn.released.connect(self._show_after)
        bottom_toolbar.addWidget(self._before_btn)
        bottom_toolbar.addSeparator()

        self._apply_btn = QToolButton()
        self._apply_btn.setText("✓ Apply")
        self._apply_btn.setToolTip("Commit current settings to the working image (Ctrl+Return)")
        self._apply_btn.setShortcut("Ctrl+Return")
        self._apply_btn.setEnabled(False)
        self._apply_btn.setStyleSheet("QToolButton { font-weight: bold; padding: 2px 8px; }")
        self._apply_btn.clicked.connect(self._apply_working)
        bottom_toolbar.addWidget(self._apply_btn)
        bottom_toolbar.addSeparator()

        # ── Connections ───────────────────────────────────────────────────────
        self.open_btn.clicked.connect(self.open_image)
        self.save_btn.clicked.connect(self.save_result)

        self.scale_slider.valueChanged.connect(self.scale_spin.setValue)
        self.scale_spin.valueChanged.connect(self.scale_slider.setValue)
        self.scale_spin.valueChanged.connect(self.schedule_preview)

        self.blur_slider.valueChanged.connect(lambda v: self.blur_spin.setValue(v / 2.0))
        self.blur_spin.valueChanged.connect(lambda v: self.blur_slider.setValue(int(v * 2)))
        self.blur_spin.valueChanged.connect(self.schedule_preview)

        self.colors_spin.valueChanged.connect(self.schedule_preview)
        self.dither_check.toggled.connect(self.schedule_preview)
        self.dither_method_combo.currentIndexChanged.connect(self.schedule_preview)
        self.resample_combo.currentIndexChanged.connect(self.schedule_preview)

        self._update_dither_controls()
        self.setMinimumSize(1100, 700)
        self.resize(1100, 700)
        self.statusBar().showMessage("")

        # Apply neutral defaults now that all widgets, timer, and canvas exist.
        self._apply_emissive_params(_get_emissive().EMISSIVE_NEUTRAL, trigger_preview=False)

    # ── Worker pre-warm ───────────────────────────────────────────────────────

    def _prewarm_worker(self):
        """Submit a no-op to force the worker subprocess to start immediately.

        This runs in a daemon thread so it doesn't block __init__ or the Qt loop.
        The subprocess spawn + import cost (~200–400 ms) is paid here, in the
        background, rather than on the user's first preview click.
        """
        try:
            future = self._process_pool.submit(_noop_worker)
            future.result(timeout=30)   # wait quietly; ignore result
        except Exception:
            pass   # pre-warm failure is non-fatal

    # ── Dither controls ───────────────────────────────────────────────────────

    def _update_dither_controls(self):
        has_sklearn = _get_processing().sklearn_available()
        if not has_sklearn:
            self.dither_method_combo.setCurrentIndex(1)
        self.dither_method_combo.setEnabled(has_sklearn)

    # ── Emissive tab builder ──────────────────────────────────────────────────

    def _build_emissive_tab(self):

        inner = QWidget()
        inner_layout = QVBoxLayout(inner)
        inner_layout.setContentsMargins(4, 4, 4, 4)
        inner_layout.setSpacing(6)

        def _slider_row(label, slider, spin):
            row = QWidget()
            rl = QHBoxLayout(row)
            rl.setContentsMargins(0, 0, 0, 0)
            rl.addWidget(QLabel(label))
            rl.addWidget(slider)
            rl.addWidget(spin)
            return row

        # Preset row
        self._emissive_presets: list[dict] = _load_emissive_presets()
        preset_row = QWidget()
        preset_layout = QHBoxLayout(preset_row)
        preset_layout.setContentsMargins(0, 0, 0, 0)
        preset_layout.addWidget(QLabel("Preset"))
        self._em_preset_combo = QComboBox()
        self._em_preset_combo.addItem("Default")          # index 0 — always clean slate
        for preset in self._emissive_presets:
            self._em_preset_combo.addItem(preset["_name"])
        self._em_preset_combo.currentIndexChanged.connect(self._on_emissive_preset_changed)
        preset_layout.addWidget(self._em_preset_combo)
        inner_layout.addWidget(preset_row)

        # --- 1. Levels group ---
        levels_grp = CollapsibleGroupBox("Levels")
        self._em_shadow_slider = QSlider(Qt.Horizontal); self._em_shadow_slider.setRange(0, 127); self._em_shadow_slider.setValue(0)
        self._em_shadow_spin = QSpinBox(); self._em_shadow_spin.setRange(0, 127); self._em_shadow_spin.setValue(0)
        self._em_shadow_slider.valueChanged.connect(self._em_shadow_spin.setValue)
        self._em_shadow_spin.valueChanged.connect(self._em_shadow_slider.setValue)
        self._em_shadow_spin.valueChanged.connect(self._mark_emissive_custom)
        self._em_shadow_spin.valueChanged.connect(self.schedule_preview)
        levels_grp.addWidget(_slider_row("Shadow", self._em_shadow_slider, self._em_shadow_spin))

        self._em_highlight_slider = QSlider(Qt.Horizontal); self._em_highlight_slider.setRange(128, 255); self._em_highlight_slider.setValue(255)
        self._em_highlight_spin = QSpinBox(); self._em_highlight_spin.setRange(128, 255); self._em_highlight_spin.setValue(255)
        self._em_highlight_slider.valueChanged.connect(self._em_highlight_spin.setValue)
        self._em_highlight_spin.valueChanged.connect(self._em_highlight_slider.setValue)
        self._em_highlight_spin.valueChanged.connect(self._mark_emissive_custom)
        self._em_highlight_spin.valueChanged.connect(self.schedule_preview)
        levels_grp.addWidget(_slider_row("Highlight", self._em_highlight_slider, self._em_highlight_spin))

        self._em_gamma_spin = QDoubleSpinBox(); self._em_gamma_spin.setRange(0.1, 5.0); self._em_gamma_spin.setSingleStep(0.1); self._em_gamma_spin.setDecimals(2); self._em_gamma_spin.setValue(1.0)
        self._em_gamma_spin.valueChanged.connect(self._mark_emissive_custom)
        self._em_gamma_spin.valueChanged.connect(self.schedule_preview)
        levels_grp.addWidget(QLabel("Gamma")); levels_grp.addWidget(self._em_gamma_spin)
        inner_layout.addWidget(levels_grp)

        # --- 2. Color grading group ---
        color_grp = CollapsibleGroupBox("Color Grading")
        self._em_color_mode = QComboBox(); self._em_color_mode.addItems(["None", "Gradient Map", "Colorize"])
        self._em_color_mode.currentIndexChanged.connect(self._mark_emissive_custom)
        self._em_color_mode.currentIndexChanged.connect(self._update_color_mode_visibility)
        self._em_color_mode.currentIndexChanged.connect(self.schedule_preview)
        color_grp.addWidget(QLabel("Mode")); color_grp.addWidget(self._em_color_mode)

        self._gm_widget = QWidget()
        gm_layout = QVBoxLayout(self._gm_widget); gm_layout.setContentsMargins(0,0,0,0)
        self._em_gm_shadow_btn = ColorButton((0, 0, 0)); self._em_gm_shadow_btn.color_changed.connect(self._mark_emissive_custom); self._em_gm_shadow_btn.color_changed.connect(self.schedule_preview)
        self._em_gm_mid_btn = ColorButton((128, 128, 128)); self._em_gm_mid_btn.color_changed.connect(self._mark_emissive_custom); self._em_gm_mid_btn.color_changed.connect(self.schedule_preview)
        self._em_gm_highlight_btn = ColorButton((255, 255, 255)); self._em_gm_highlight_btn.color_changed.connect(self._mark_emissive_custom); self._em_gm_highlight_btn.color_changed.connect(self.schedule_preview)
        gm_layout.addWidget(QLabel("Shadow")); gm_layout.addWidget(self._em_gm_shadow_btn)
        gm_layout.addWidget(QLabel("Midtone")); gm_layout.addWidget(self._em_gm_mid_btn)
        gm_layout.addWidget(QLabel("Highlight")); gm_layout.addWidget(self._em_gm_highlight_btn)
        color_grp.addWidget(self._gm_widget)

        self._colorize_widget = QWidget()
        cz_layout = QVBoxLayout(self._colorize_widget); cz_layout.setContentsMargins(0,0,0,0)
        self._em_hue_spin = QSpinBox(); self._em_hue_spin.setRange(0, 360); self._em_hue_spin.setValue(0)
        self._em_hue_spin.valueChanged.connect(self._mark_emissive_custom); self._em_hue_spin.valueChanged.connect(self.schedule_preview)
        self._em_sat_spin = QSpinBox(); self._em_sat_spin.setRange(0, 100); self._em_sat_spin.setValue(0)
        self._em_sat_spin.valueChanged.connect(self._mark_emissive_custom); self._em_sat_spin.valueChanged.connect(self.schedule_preview)
        cz_layout.addWidget(QLabel("Hue")); cz_layout.addWidget(self._em_hue_spin)
        cz_layout.addWidget(QLabel("Saturation")); cz_layout.addWidget(self._em_sat_spin)
        color_grp.addWidget(self._colorize_widget)
        inner_layout.addWidget(color_grp)
        self._update_color_mode_visibility()

        # --- 3. Glow group ---
        glow_grp = CollapsibleGroupBox("Emissive Glow")
        self._em_glow_check = QCheckBox("Enable"); self._em_glow_check.setChecked(False)
        self._em_glow_check.toggled.connect(self._mark_emissive_custom); self._em_glow_check.toggled.connect(self.schedule_preview)
        glow_grp.addWidget(self._em_glow_check)

        self._em_glow_blur_slider = QSlider(Qt.Horizontal); self._em_glow_blur_slider.setRange(1, 100); self._em_glow_blur_slider.setValue(10)
        self._em_glow_blur_spin = QSpinBox(); self._em_glow_blur_spin.setRange(1, 100); self._em_glow_blur_spin.setValue(10)
        self._em_glow_blur_slider.valueChanged.connect(self._em_glow_blur_spin.setValue)
        self._em_glow_blur_spin.valueChanged.connect(self._em_glow_blur_slider.setValue)
        self._em_glow_blur_spin.valueChanged.connect(self._mark_emissive_custom); self._em_glow_blur_spin.valueChanged.connect(self.schedule_preview)
        glow_grp.addWidget(_slider_row("Blur", self._em_glow_blur_slider, self._em_glow_blur_spin))

        self._em_glow_blend = QComboBox(); self._em_glow_blend.addItems(["Screen", "Linear Dodge", "Color Dodge"])
        self._em_glow_blend.currentIndexChanged.connect(self._mark_emissive_custom); self._em_glow_blend.currentIndexChanged.connect(self.schedule_preview)
        glow_grp.addWidget(QLabel("Blend")); glow_grp.addWidget(self._em_glow_blend)

        self._em_glow_opacity_slider = QSlider(Qt.Horizontal); self._em_glow_opacity_slider.setRange(0, 100); self._em_glow_opacity_slider.setValue(50)
        self._em_glow_opacity_spin = QSpinBox(); self._em_glow_opacity_spin.setRange(0, 100); self._em_glow_opacity_spin.setValue(50); self._em_glow_opacity_spin.setSuffix(" %")
        self._em_glow_opacity_slider.valueChanged.connect(self._em_glow_opacity_spin.setValue)
        self._em_glow_opacity_spin.valueChanged.connect(self._em_glow_opacity_slider.setValue)
        self._em_glow_opacity_spin.valueChanged.connect(self._mark_emissive_custom); self._em_glow_opacity_spin.valueChanged.connect(self.schedule_preview)
        glow_grp.addWidget(_slider_row("Opacity", self._em_glow_opacity_slider, self._em_glow_opacity_spin))

        self._em_glow_thresh_slider = QSlider(Qt.Horizontal); self._em_glow_thresh_slider.setRange(0, 255); self._em_glow_thresh_slider.setValue(128)
        self._em_glow_thresh_spin = QSpinBox(); self._em_glow_thresh_spin.setRange(0, 255); self._em_glow_thresh_spin.setValue(128)
        self._em_glow_thresh_slider.valueChanged.connect(self._em_glow_thresh_spin.setValue)
        self._em_glow_thresh_spin.valueChanged.connect(self._em_glow_thresh_slider.setValue)
        self._em_glow_thresh_spin.valueChanged.connect(self._mark_emissive_custom); self._em_glow_thresh_spin.valueChanged.connect(self.schedule_preview)
        glow_grp.addWidget(_slider_row("Threshold", self._em_glow_thresh_slider, self._em_glow_thresh_spin))
        inner_layout.addWidget(glow_grp)

        # --- 4. Softness group ---
        soft_grp = CollapsibleGroupBox("Dream Softness")
        self._em_soft_check = QCheckBox("Enable"); self._em_soft_check.setChecked(False)
        self._em_soft_check.toggled.connect(self._mark_emissive_custom); self._em_soft_check.toggled.connect(self.schedule_preview)
        soft_grp.addWidget(self._em_soft_check)

        self._em_soft_blur_slider = QSlider(Qt.Horizontal); self._em_soft_blur_slider.setRange(5, 120); self._em_soft_blur_slider.setValue(20)
        self._em_soft_blur_spin = QDoubleSpinBox(); self._em_soft_blur_spin.setRange(0.5, 12.0); self._em_soft_blur_spin.setSingleStep(0.5); self._em_soft_blur_spin.setDecimals(1); self._em_soft_blur_spin.setValue(2.0)
        self._em_soft_blur_slider.valueChanged.connect(lambda v: self._em_soft_blur_spin.setValue(v / 10.0))
        self._em_soft_blur_spin.valueChanged.connect(lambda v: self._em_soft_blur_slider.setValue(int(v * 10)))
        self._em_soft_blur_spin.valueChanged.connect(self._mark_emissive_custom); self._em_soft_blur_spin.valueChanged.connect(self.schedule_preview)
        soft_grp.addWidget(_slider_row("Blur", self._em_soft_blur_slider, self._em_soft_blur_spin))

        self._em_soft_blend = QComboBox(); self._em_soft_blend.addItems(["Soft Light", "Overlay"])
        self._em_soft_blend.currentIndexChanged.connect(self._mark_emissive_custom); self._em_soft_blend.currentIndexChanged.connect(self.schedule_preview)
        soft_grp.addWidget(QLabel("Blend")); soft_grp.addWidget(self._em_soft_blend)

        self._em_soft_opacity_slider = QSlider(Qt.Horizontal); self._em_soft_opacity_slider.setRange(0, 50); self._em_soft_opacity_slider.setValue(20)
        self._em_soft_opacity_spin = QSpinBox(); self._em_soft_opacity_spin.setRange(0, 50); self._em_soft_opacity_spin.setValue(20); self._em_soft_opacity_spin.setSuffix(" %")
        self._em_soft_opacity_slider.valueChanged.connect(self._em_soft_opacity_spin.setValue)
        self._em_soft_opacity_spin.valueChanged.connect(self._em_soft_opacity_slider.setValue)
        self._em_soft_opacity_spin.valueChanged.connect(self._mark_emissive_custom); self._em_soft_opacity_spin.valueChanged.connect(self.schedule_preview)
        soft_grp.addWidget(_slider_row("Opacity", self._em_soft_opacity_slider, self._em_soft_opacity_spin))
        inner_layout.addWidget(soft_grp)

        # --- 5. Color Bloom group ---
        bloom_grp = CollapsibleGroupBox("Color Bloom", collapsed=True)
        self._em_bloom_check = QCheckBox("Enable"); self._em_bloom_check.setChecked(False)
        self._em_bloom_check.toggled.connect(self._mark_emissive_custom); self._em_bloom_check.toggled.connect(self.schedule_preview)
        bloom_grp.addWidget(self._em_bloom_check)

        self._em_bloom_color_btn = ColorButton((255, 255, 255))
        self._em_bloom_color_btn.color_changed.connect(self._mark_emissive_custom); self._em_bloom_color_btn.color_changed.connect(self.schedule_preview)
        bloom_grp.addWidget(QLabel("Color")); bloom_grp.addWidget(self._em_bloom_color_btn)

        self._em_bloom_blend = QComboBox(); self._em_bloom_blend.addItems(["Overlay", "Soft Light"])
        self._em_bloom_blend.currentIndexChanged.connect(self._mark_emissive_custom); self._em_bloom_blend.currentIndexChanged.connect(self.schedule_preview)
        bloom_grp.addWidget(QLabel("Blend")); bloom_grp.addWidget(self._em_bloom_blend)

        self._em_bloom_opacity_slider = QSlider(Qt.Horizontal); self._em_bloom_opacity_slider.setRange(0, 50); self._em_bloom_opacity_slider.setValue(20)
        self._em_bloom_opacity_spin = QSpinBox(); self._em_bloom_opacity_spin.setRange(0, 50); self._em_bloom_opacity_spin.setValue(20); self._em_bloom_opacity_spin.setSuffix(" %")
        self._em_bloom_opacity_slider.valueChanged.connect(self._em_bloom_opacity_spin.setValue)
        self._em_bloom_opacity_spin.valueChanged.connect(self._em_bloom_opacity_slider.setValue)
        self._em_bloom_opacity_spin.valueChanged.connect(self._mark_emissive_custom); self._em_bloom_opacity_spin.valueChanged.connect(self.schedule_preview)
        bloom_grp.addWidget(_slider_row("Opacity", self._em_bloom_opacity_slider, self._em_bloom_opacity_spin))
        inner_layout.addWidget(bloom_grp)

        # --- 6. Grain group ---
        grain_grp = CollapsibleGroupBox("Grain")
        self._em_grain_check = QCheckBox("Enable"); self._em_grain_check.setChecked(False)
        self._em_grain_check.toggled.connect(self._mark_emissive_custom); self._em_grain_check.toggled.connect(self.schedule_preview)
        grain_grp.addWidget(self._em_grain_check)

        self._em_grain_slider = QSlider(Qt.Horizontal); self._em_grain_slider.setRange(0, 100); self._em_grain_slider.setValue(20)
        self._em_grain_spin = QSpinBox(); self._em_grain_spin.setRange(0, 100); self._em_grain_spin.setValue(20); self._em_grain_spin.setSuffix(" %")
        self._em_grain_slider.valueChanged.connect(self._em_grain_spin.setValue)
        self._em_grain_spin.valueChanged.connect(self._em_grain_slider.setValue)
        self._em_grain_spin.valueChanged.connect(self._mark_emissive_custom); self._em_grain_spin.valueChanged.connect(self.schedule_preview)
        grain_grp.addWidget(_slider_row("Intensity", self._em_grain_slider, self._em_grain_spin))

        self._em_grain_type = QComboBox(); self._em_grain_type.addItems(["Gaussian", "Uniform"])
        self._em_grain_type.currentIndexChanged.connect(self._mark_emissive_custom); self._em_grain_type.currentIndexChanged.connect(self.schedule_preview)
        grain_grp.addWidget(QLabel("Type")); grain_grp.addWidget(self._em_grain_type)
        inner_layout.addWidget(grain_grp)
        inner_layout.addStretch()

        em_reset_btn = QPushButton("\u21ba  Reset")
        em_reset_btn.setToolTip("Reset all emissive parameters to neutral defaults")
        em_reset_btn.clicked.connect(self._reset_emissive_params)
        inner_layout.addWidget(em_reset_btn)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(inner)
        scroll.setFrameShape(QFrame.NoFrame)
        self._right_tabs.addTab(scroll, "Emissive Dreamy")

    # ── Tab reset helpers ─────────────────────────────────────────────────────

    def _reset_quant_params(self):
        """Reset all Quantization tab controls to their startup defaults."""
        widgets = [
            self.scale_slider, self.scale_spin,
            self.blur_slider, self.blur_spin,
            self.colors_spin, self.dither_check,
            self.dither_method_combo, self.resample_combo,
        ]
        for w in widgets:
            w.blockSignals(True)

        self.scale_slider.setValue(100)
        self.scale_spin.setValue(100)
        self.blur_slider.setValue(0)
        self.blur_spin.setValue(0.0)
        self.colors_spin.setValue(256)
        self.dither_check.setChecked(False)
        self.dither_method_combo.setCurrentIndex(0)
        self.resample_combo.setCurrentIndex(0)   # Nearest — widget startup default

        for w in widgets:
            w.blockSignals(False)
        self._update_dither_controls()   # re-apply sklearn availability constraint
        self.schedule_preview()

    def _reset_quant_params_silent(self):
        """Reset quantization controls to defaults without triggering a preview."""
        widgets = [
            self.scale_slider, self.scale_spin,
            self.blur_slider, self.blur_spin,
            self.colors_spin, self.dither_check,
            self.dither_method_combo, self.resample_combo,
        ]
        for w in widgets:
            w.blockSignals(True)

        self.scale_slider.setValue(100)
        self.scale_spin.setValue(100)
        self.blur_slider.setValue(0)
        self.blur_spin.setValue(0.0)
        self.colors_spin.setValue(256)
        self.dither_check.setChecked(False)
        self.dither_method_combo.setCurrentIndex(0)
        self.resample_combo.setCurrentIndex(0)

        for w in widgets:
            w.blockSignals(False)
        self._update_dither_controls()

    def _reset_emissive_params_silent(self):
        """Reset emissive controls to neutral defaults without triggering a preview."""
        self._em_preset_combo.blockSignals(True)
        self._em_preset_combo.setCurrentIndex(0)
        self._em_preset_combo.blockSignals(False)
        self._apply_emissive_params(_get_emissive().EMISSIVE_NEUTRAL, trigger_preview=False)

    def _reset_emissive_params(self):
        """Reset all Emissive Dreamy controls to neutral and set combo to Default."""
        self._em_preset_combo.blockSignals(True)
        self._em_preset_combo.setCurrentIndex(0)
        self._em_preset_combo.blockSignals(False)
        self._apply_emissive_params(_get_emissive().EMISSIVE_NEUTRAL)
        """Reset all Emissive Dreamy controls to neutral and set combo to Default."""
        self._em_preset_combo.blockSignals(True)
        self._em_preset_combo.setCurrentIndex(0)
        self._em_preset_combo.blockSignals(False)
        self._apply_emissive_params(_get_emissive().EMISSIVE_NEUTRAL)

    # ── Emissive helpers ──────────────────────────────────────────────────────

    def _update_color_mode_visibility(self, _idx=None):
        idx = self._em_color_mode.currentIndex()
        self._gm_widget.setVisible(idx == 1)
        self._colorize_widget.setVisible(idx == 2)

    def _mark_emissive_custom(self, *_args):
        if self._em_preset_combo.currentIndex() != 0:
            self._em_preset_combo.blockSignals(True)
            self._em_preset_combo.setCurrentIndex(0)  # back to "Default" (free edit)
            self._em_preset_combo.blockSignals(False)

    def _on_emissive_preset_changed(self, idx: int):
        if idx == 0:
            self._apply_emissive_params(_get_emissive().EMISSIVE_NEUTRAL)
        else:
            preset = self._emissive_presets[idx - 1]   # offset by 1 for "Default" slot
            self._apply_emissive_params(preset)

    def _apply_emissive_params(self, params: dict, trigger_preview: bool = True):
        """Push any dict of emissive parameter values onto the UI widgets.

        Only the keys present in *params* are applied; everything else keeps its
        current widget value.  This means you can pass a sparse preset (e.g. only
        ``glow_enabled``) and all unmentioned controls are untouched.

        Adding a new parameter to the pipeline never requires changing this method —
        just add the key to the widget-map below.
        """
        # Map: param key → callable that sets the widget value
        # Entries are (setter,) for write-only, or (blocker_widget, setter) when
        # the same widget should have signals blocked while being set.
        def _color_mode_index(mode: str) -> int:
            return {"none": 0, "gradient_map": 1, "colorize": 2}.get(mode, 0)

        def _combo_text(combo: QComboBox, text: str):
            combo.setCurrentText(text)

        widget_setters = {
            "levels_shadow":    lambda v: self._em_shadow_spin.setValue(int(v)),
            "levels_highlight": lambda v: self._em_highlight_spin.setValue(int(v)),
            "levels_gamma":     lambda v: self._em_gamma_spin.setValue(float(v)),
            "color_mode":       lambda v: self._em_color_mode.setCurrentIndex(_color_mode_index(v)),
            "gm_shadow":        lambda v: self._em_gm_shadow_btn.set_color(tuple(v)),
            "gm_mid":           lambda v: self._em_gm_mid_btn.set_color(tuple(v)),
            "gm_highlight":     lambda v: self._em_gm_highlight_btn.set_color(tuple(v)),
            "colorize_hue":     lambda v: self._em_hue_spin.setValue(int(v)),
            "colorize_sat":     lambda v: self._em_sat_spin.setValue(int(v)),
            "glow_enabled":     lambda v: self._em_glow_check.setChecked(bool(v)),
            "glow_blur":        lambda v: self._em_glow_blur_spin.setValue(int(v)),
            "glow_blend":       lambda v: _combo_text(self._em_glow_blend, v),
            "glow_opacity":     lambda v: self._em_glow_opacity_spin.setValue(int(float(v) * 100)),
            "glow_threshold":   lambda v: self._em_glow_thresh_spin.setValue(int(v)),
            "soft_enabled":     lambda v: self._em_soft_check.setChecked(bool(v)),
            "soft_blur":        lambda v: self._em_soft_blur_spin.setValue(float(v)),
            "soft_blend":       lambda v: _combo_text(self._em_soft_blend, v),
            "soft_opacity":     lambda v: self._em_soft_opacity_spin.setValue(int(float(v) * 100)),
            "bloom_enabled":    lambda v: self._em_bloom_check.setChecked(bool(v)),
            "bloom_color":      lambda v: self._em_bloom_color_btn.set_color(tuple(v)),
            "bloom_blend":      lambda v: _combo_text(self._em_bloom_blend, v),
            "bloom_opacity":    lambda v: self._em_bloom_opacity_spin.setValue(int(float(v) * 100)),
            "grain_enabled":    lambda v: self._em_grain_check.setChecked(bool(v)),
            "grain_intensity":  lambda v: self._em_grain_spin.setValue(int(float(v) * 100)),
            "grain_type":       lambda v: _combo_text(self._em_grain_type, str(v).capitalize()),
        }

        # Collect every widget that will be touched so we can block/unblock in bulk
        all_widgets = [
            self._em_shadow_spin, self._em_shadow_slider,
            self._em_highlight_spin, self._em_highlight_slider,
            self._em_gamma_spin, self._em_color_mode,
            self._em_gm_shadow_btn, self._em_gm_mid_btn, self._em_gm_highlight_btn,
            self._em_hue_spin, self._em_sat_spin,
            self._em_glow_check, self._em_glow_blur_spin, self._em_glow_blend,
            self._em_glow_opacity_spin, self._em_glow_thresh_spin,
            self._em_soft_check, self._em_soft_blur_spin, self._em_soft_blend,
            self._em_soft_opacity_spin,
            self._em_bloom_check, self._em_bloom_color_btn, self._em_bloom_blend,
            self._em_bloom_opacity_spin,
            self._em_grain_check, self._em_grain_spin, self._em_grain_type,
        ]
        for w in all_widgets:
            w.blockSignals(True)

        for key, setter in widget_setters.items():
            if key in params:
                try:
                    setter(params[key])
                except Exception as exc:
                    print(f"[presets] failed to apply '{key}': {exc}")

        for w in all_widgets:
            w.blockSignals(False)

        self._update_color_mode_visibility()
        if trigger_preview:
            self.schedule_preview()

    def _collect_emissive_params(self) -> dict:
        return {
            "levels_shadow":    self._em_shadow_spin.value(),
            "levels_highlight": self._em_highlight_spin.value(),
            "levels_gamma":     self._em_gamma_spin.value(),
            "color_mode":       {0: "none", 1: "gradient_map", 2: "colorize"}.get(self._em_color_mode.currentIndex(), "none"),
            "gm_shadow":        self._em_gm_shadow_btn.color(),
            "gm_mid":           self._em_gm_mid_btn.color(),
            "gm_highlight":     self._em_gm_highlight_btn.color(),
            "colorize_hue":     self._em_hue_spin.value(),
            "colorize_sat":     self._em_sat_spin.value(),
            "glow_enabled":     self._em_glow_check.isChecked(),
            "glow_blur":        float(self._em_glow_blur_spin.value()),
            "glow_blend":       self._em_glow_blend.currentText(),
            "glow_opacity":     self._em_glow_opacity_spin.value() / 100.0,
            "glow_threshold":   self._em_glow_thresh_spin.value(),
            "soft_enabled":     self._em_soft_check.isChecked(),
            "soft_blur":        self._em_soft_blur_spin.value(),
            "soft_blend":       self._em_soft_blend.currentText(),
            "soft_opacity":     self._em_soft_opacity_spin.value() / 100.0,
            "bloom_enabled":    self._em_bloom_check.isChecked(),
            "bloom_color":      self._em_bloom_color_btn.color(),
            "bloom_blend":      self._em_bloom_blend.currentText(),
            "bloom_opacity":    self._em_bloom_opacity_spin.value() / 100.0,
            "grain_enabled":    self._em_grain_check.isChecked(),
            "grain_intensity":  self._em_grain_spin.value() / 100.0,
            "grain_type":       self._em_grain_type.currentText().lower(),
        }

    # ── Preview scheduling ────────────────────────────────────────────────────

    def schedule_preview(self, *_):
        self._preview_timer.start()

    def _do_preview(self):
        self.apply_processing()

    # ── Image I/O ─────────────────────────────────────────────────────────────

    def open_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open image")
        if not path:
            return
        Image = _get_PIL_Image()
        ImageQt = _get_PIL_ImageQt()
        try:
            self.image = Image.open(path)
        except Exception:
            self.image = None
            return
        self._source_image = self.image.convert("RGBA").copy()
        self._working_image = self._source_image.copy()
        self._preview_image = None
        self._undo_stack.clear()
        self._redo_stack.clear()
        self._apply_btn.setEnabled(True)
        self._update_undo_redo_actions()
        pix = QPixmap.fromImage(ImageQt(self._working_image))
        self._current_pixmap = pix
        self.preview.set_pixmap(pix, ref_size=self._working_image.size)

        # Reset both tabs to their defaults whenever a new image is opened,
        # but suppress the per-widget preview signals and skip the auto-commit
        # timer — the freshly loaded image is already displayed above.
        self._reset_quant_params_silent()
        self._reset_emissive_params_silent()

    def show_pil(self, pil_img):
        """Display a PIL image, preserving current zoom and pan."""
        ImageQt = _get_PIL_ImageQt()
        qim = ImageQt(pil_img.convert("RGBA"))
        pix = QPixmap.fromImage(qim)
        self._current_pixmap = pix
        self.preview.set_pixmap_keep_view(pix)

    def resizeEvent(self, event):
        super().resizeEvent(event)

    # ── Zoom ──────────────────────────────────────────────────────────────────

    def eventFilter(self, obj, event):
        if obj is self.preview and isinstance(event, QWheelEvent):
            if event.modifiers() & Qt.ControlModifier:
                delta = event.angleDelta().y()
                factor = 1.15 if delta > 0 else (1 / 1.15)
                anchor = QPointF(event.position())
                new_zoom = self.preview.apply_zoom(factor, anchor)
                if new_zoom is not None:
                    self.statusBar().showMessage(f"Zoom: {new_zoom * 100:.0f}%", 2000)
                return True
        return super().eventFilter(obj, event)

    def _zoom_fit(self):
        self.preview.reset_view()
        self.statusBar().showMessage("Fit to window", 1500)

    # ── Before / After ────────────────────────────────────────────────────────

    def _show_before(self):
        if self.image is None:
            return
        self._showing_original = True
        self.show_pil(self.image)

    def _on_tab_changed(self, _idx: int):
        if self._working_image is None:
            return
        # Show whatever was last displayed (preview or committed) while the
        # new tab's worker job is in flight, then schedule a fresh preview
        # using the new tab's current parameters.
        img = self._preview_image if self._preview_image is not None else self._working_image
        self.show_pil(img)
        self._preview_image = None
        self.schedule_preview()

    def _show_after(self):
        self._showing_original = False
        img = self._preview_image if self._preview_image is not None else self._working_image
        if img is not None:
            self.show_pil(img)

    # ── Processing ────────────────────────────────────────────────────────────

    def _is_emissive_mode(self) -> bool:
        return self._right_tabs.currentIndex() == 1

    def _quant_is_passthrough(self) -> bool:
        """Return True when all quantization controls are at their neutral values.

        At neutral the pipeline produces a pixel-identical result, so we can
        skip the worker entirely and show the working image directly.
        """
        return (
            self.scale_spin.value() == 100
            and self.blur_spin.value() == 0.0
            and self.colors_spin.value() == 256
            and not self.dither_check.isChecked()
        )

    def _build_quant_inputs(self) -> tuple:
        Image = _get_PIL_Image()
        ImageFilter = _get_PIL_ImageFilter()
        scale = self.scale_spin.value() / 100.0
        blur  = self.blur_spin.value()
        colors = self.colors_spin.value()
        dither = self.dither_check.isChecked()
        use_pillow_dither = self.dither_method_combo.currentIndex() == 1
        resample_idx = self.resample_combo.currentIndex()
        # Resolve the string name to the actual Image constant lazily
        resample_name = self.resample_map.get(resample_idx, "LANCZOS")
        resample_filter = getattr(Image, resample_name, Image.LANCZOS)
        img = self._source_image.copy()
        if scale != 1.0:
            w, h = img.size
            new_size = (max(1, int(w * scale)), max(1, int(h * scale)))
            img = img.resize(new_size, resample=resample_filter)
        img = img if blur <= 0 else img.filter(ImageFilter.GaussianBlur(blur))
        return img, {
            "colors": colors,
            "dither": dither,
            "use_pillow_dither": use_pillow_dither,
            "passthrough": self._quant_is_passthrough(),
        }

    def apply_processing(self):
        if self._working_image is None:
            return
        if self._is_emissive_mode():
            self._start_worker_generic(
                "emissive", self._working_image.copy(),
                self._collect_emissive_params(), draft=True, commit=False,
            )
        else:
            if self._quant_is_passthrough():
                # Nothing to do — show the working image pixel-perfect.
                self._preview_image = None
                self.show_pil(self._working_image)
                return
            img, params = self._build_quant_inputs()
            self._start_worker_generic("quantize", img, params, draft=False, commit=False)

    def _apply_working(self):
        if self._source_image is None:
            return
        self._apply_btn.setEnabled(False)
        self.statusBar().showMessage("Applying…")
        if self._is_emissive_mode():
            self._start_worker_generic(
                "emissive", self._source_image.copy(),
                self._collect_emissive_params(), draft=False, commit=True,
            )
        else:
            img, params = self._build_quant_inputs()
            self._start_worker_generic("quantize", img, params, draft=False, commit=True)

    def _undo(self):
        if not self._undo_stack:
            return
        self._redo_stack.append(self._working_image)
        self._working_image = self._undo_stack.pop()
        self._preview_image = None
        self.show_pil(self._working_image)
        self._update_undo_redo_actions()
        self.statusBar().showMessage(f"Undo — {len(self._undo_stack)} step(s) remaining", 2000)

    def _redo(self):
        if not self._redo_stack:
            return
        self._undo_stack.append(self._working_image)
        self._working_image = self._redo_stack.pop()
        self._preview_image = None
        self.show_pil(self._working_image)
        self._update_undo_redo_actions()
        self.statusBar().showMessage(f"Redo — {len(self._undo_stack)} step(s) in history", 2000)

    def _update_undo_redo_actions(self):
        self._undo_action.setEnabled(bool(self._undo_stack))
        self._redo_action.setEnabled(bool(self._redo_stack))

    def _start_worker_generic(self, mode, img, params, draft=False, commit=False):
        if self._current_future is not None and not self._current_future.done():
            self._pending_params = (mode, img, params, draft, commit)
            return
        self._current_job_commit = commit
        if not commit:
            self.statusBar().showMessage("Preview…")
        ipc = _get_ipc()
        img_bytes = ipc._img_to_bytes(img)
        future = self._process_pool.submit(ipc._worker_process, mode, img_bytes, params, draft)
        self._current_future = future
        future.add_done_callback(lambda f: self.processing_done.emit(f))

    def save_result(self):
        target = self._working_image
        if target is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save result", filter="PNG Files (*.png);;All Files (*)")
        if not path:
            return
        try:
            target.save(path)
            self.statusBar().showMessage("Saved!", 2000)
        except Exception as exc:
            self.statusBar().showMessage(f"Save failed: {exc}", 3000)

    def _on_future_done(self, future: concurrent.futures.Future):
        print("[gui] Future done callback invoked")
        try:
            data = future.result()
        except Exception as exc:
            tb = traceback.format_exception(type(exc), exc, exc.__traceback__)
            print("Background worker exception:\n", "".join(tb))
            try:
                self.statusBar().showMessage(f"Processing failed: {exc}")
            except Exception:
                pass
            data = None

        if isinstance(data, (bytes, bytearray)):
            try:
                data = _get_ipc()._bytes_to_img(data)
            except Exception as exc2:
                print("Failed to decode image bytes from worker:", exc2)
                data = None

        Image = _get_PIL_Image()
        if data:
            try:
                if isinstance(data, Image.Image):
                    if self._current_job_commit:
                        if self._working_image is not None:
                            self._undo_stack.append(self._working_image)
                        self._redo_stack.clear()
                        self._working_image = data
                        self._preview_image = None
                        self._apply_btn.setEnabled(True)
                        self._update_undo_redo_actions()
                        self.statusBar().showMessage(
                            f"Applied  —  {len(self._undo_stack)} step(s) in history", 3000)
                    else:
                        self._preview_image = data
                    self.show_pil(data)
            except Exception as exc:
                print("Failed to set result image:", exc)
                if self._current_job_commit:
                    self._apply_btn.setEnabled(True)
        else:
            if self._current_job_commit:
                self._apply_btn.setEnabled(True)

        if not self._current_job_commit:
            self.statusBar().clearMessage()
        self._current_future = None
        self._current_job_commit = False
        self._update_dither_controls()

        if self._pending_params is not None:
            params = self._pending_params
            self._pending_params = None
            self._start_worker_generic(*params)

    # ── About / Settings ──────────────────────────────────────────────────────

    def _show_about(self):
        QMessageBox.about(
            self, "About",
            "<b>Python Prototype</b><br><br>"
            "An image processing tool.<br><br>"
            "Built with PySide6 and Pillow.",
        )

    def _show_settings(self):
        dlg = QDialog(self)
        dlg.setWindowTitle("Settings")
        dlg.setMinimumWidth(320)
        layout = QVBoxLayout(dlg)
        layout.addWidget(QLabel("Settings will be added here."))
        buttons = QDialogButtonBox(QDialogButtonBox.Ok)
        buttons.accepted.connect(dlg.accept)
        layout.addWidget(buttons)
        dlg.exec()

    # ── Close ─────────────────────────────────────────────────────────────────

    def closeEvent(self, event):
        try:
            if self._current_future is not None and not self._current_future.done():
                try:
                    self.statusBar().showMessage("Cancelling background processing...")
                    self._current_future.cancel()
                except Exception:
                    pass
            try:
                self._process_pool.shutdown(wait=False)
            except Exception:
                pass
            try:
                procs = getattr(self._process_pool, "_processes", {})
                for p in list(procs.values()):
                    try:
                        p.terminate(); p.join(1)
                    except Exception:
                        pass
            except Exception:
                pass
        except Exception:
            pass
        super().closeEvent(event)


# ---------------------------------------------------------------------------
# No-op worker (used by pre-warm — must be a module-level picklable function)
# ---------------------------------------------------------------------------

def _noop_worker():
    """Imported by the worker subprocess to trigger its own import chain."""
    return b""


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    # Windows multiprocessing safety guard
    import multiprocessing as _mp
    _mp.freeze_support()

    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())
