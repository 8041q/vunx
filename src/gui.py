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
from PIL.ImageQt import ImageQt
from PIL import Image, ImageFilter
import sys
import traceback

from .processing import process_image, sklearn_available
from .emissive import process_emissive, EMISSIVE_DEFAULTS

import concurrent.futures
import threading


# ---------------------------------------------------------------------------
# Canvas widget — handles all pan / zoom rendering internally
# ---------------------------------------------------------------------------

class _PreviewCanvas(QWidget):
    """A widget that draws a QPixmap with pan and zoom.

    Coordinate convention
    ─────────────────────
    _zoom        : pixels of *screen* per pixel of *image*  (1.0 = 1:1)
    _pan_offset  : QPointF — top-left corner of the image in widget coords

    When _zoom <= 0 the widget is in "fit" mode and the offset is ignored;
    the image is centred and scaled to fill the widget.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.setCursor(Qt.OpenHandCursor)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumSize(200, 200)
        self.setStyleSheet("background-color: #222;")

        self._pixmap: QPixmap | None = None
        self._ref_size: tuple[int, int] | None = None  # original image dims (w, h)
        self._zoom: float = 0.0        # 0 = fit, >0 = explicit
        self._pan_offset = QPointF(0, 0)

        self._drag_active = False
        self._drag_start_mouse = QPointF()
        self._drag_start_offset = QPointF()

    # ── public API ────────────────────────────────────────────────────────────

    def set_pixmap(self, pix: QPixmap | None, ref_size: tuple[int, int] | None = None):
        """Set pixmap and reset zoom/pan to fit. Call on image open.

        ref_size: (w, h) of the original (unscaled) image. All zoom arithmetic
        is done against this size so that changing the processing scale slider
        does not affect the displayed zoom level.
        """
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
        """Swap pixmap content without touching zoom or pan.

        Use this for live-preview / processing results: the display size is
        determined by _ref_size + _zoom, so the image won't jump even if the
        new pixmap has different pixel dimensions (due to scale slider).
        """
        self._pixmap = pix
        self.update()

    def reset_view(self):
        """Return to fit mode."""
        self._zoom = 0.0
        self._pan_offset = QPointF(0, 0)
        self.update()

    def apply_zoom(self, factor: float, anchor: QPointF | None = None):
        """Zoom by *factor*, keeping the screen point *anchor* fixed."""
        if self._pixmap is None or self._ref_size is None:
            return

        old_zoom = self._effective_zoom()
        new_zoom = max(0.05, min(32.0, old_zoom * factor))
        if abs(new_zoom - old_zoom) < 1e-9:
            return

        if anchor is None:
            anchor = QPointF(self.width() / 2, self.height() / 2)

        # If we're in fit mode, seed the pan offset from the centred position
        if self._zoom <= 0:
            rw, rh = self._ref_size
            sw = rw * old_zoom
            sh = rh * old_zoom
            self._pan_offset = QPointF(
                (self.width() - sw) / 2,
                (self.height() - sh) / 2,
            )

        # point in ref-image coords under the anchor
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

    # ── rendering ─────────────────────────────────────────────────────────────

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
        # display size is always ref_size * zoom — independent of pixmap pixel count
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

    # ── mouse drag ────────────────────────────────────────────────────────────

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and self._pixmap is not None:
            self._drag_active = True
            self._drag_start_mouse = QPointF(event.position())
            if self._zoom <= 0 and self._ref_size is not None:
                # transition from fit: seed offset from the centred position
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

    # ── helpers ───────────────────────────────────────────────────────────────

    def _effective_zoom(self) -> float:
        """Return screen-pixels-per-ref-pixel zoom (fit or explicit)."""
        rw, rh = self._ref_size if self._ref_size else (
            (self._pixmap.width(), self._pixmap.height()) if self._pixmap else (1, 1)
        )
        if self._zoom > 0:
            return self._zoom
        # fit scale based on reference size
        ww, wh = self.width(), self.height()
        if rw == 0 or rh == 0:
            return 1.0
        return min(ww / rw, wh / rh)

    def _fit_offset(self, pix: QPixmap | None) -> QPointF:
        # unused now — fit mode is handled entirely in paintEvent
        return QPointF(0, 0)

    def _clamp_pan(self):
        """Prevent panning so far that the image fully leaves the viewport."""
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
    """A group box that can collapse/expand its contents with a toggle."""

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
    """A button that shows its colour and opens QColorDialog on click."""
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
        self._working_image = None    # accumulated committed state (full-res)
        self._preview_image = None    # live draft — displayed but not yet committed
        self._undo_stack: list[Image.Image] = []
        self._redo_stack: list[Image.Image] = []
        self._showing_original = False

        # Background worker state
        self._process_pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        self._current_future = None
        self._pending_params = None
        self._current_job_commit: bool = False  # True when the running job should commit on finish
        self.processing_done.connect(self._on_future_done)

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

        # Left panel
        left_group = QGroupBox("Actions")
        left_layout = QVBoxLayout()
        left_group.setLayout(left_layout)
        self.open_btn = QPushButton("Open Image")
        self.save_btn = QPushButton("Save Result")
        left_layout.addWidget(self.open_btn)
        left_layout.addWidget(self.save_btn)
        left_layout.addStretch()

        # Right panel — tab widget with Quantization + Emissive tabs
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
        self.colors_spin.setValue(16)

        self.dither_check = QCheckBox("Dither")
        self.dither_check.setChecked(True)

        self.dither_method_combo = QComboBox()
        self.dither_method_combo.addItems(["Bayer (k-means)", "Floyd–Steinberg"])

        self.resample_combo = QComboBox()
        self.resample_combo.addItems(["Nearest", "Bilinear", "Bicubic", "Lanczos"])
        self.resample_map = {
            0: Image.NEAREST, 1: Image.BILINEAR,
            2: Image.BICUBIC,  3: Image.LANCZOS,
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

        self._right_tabs.addTab(quant_widget, "Quantization")

        # ── Emissive Dreamy tab ───────────────────────────────────────────────
        self._build_emissive_tab()

        self._right_tabs.currentChanged.connect(self._on_tab_changed)

        # Canvas
        self.preview = _PreviewCanvas()
        self.preview.installEventFilter(self)   # for Ctrl+scroll

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

        self._preview_timer = QTimer(self)
        self._preview_timer.setSingleShot(True)
        self._preview_timer.setInterval(300)
        self._preview_timer.timeout.connect(self._do_preview)

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

    # ── Dither controls ───────────────────────────────────────────────────────

    def _update_dither_controls(self):
        has_sklearn = sklearn_available()
        if not has_sklearn:
            self.dither_method_combo.setCurrentIndex(1)
        self.dither_method_combo.setEnabled(has_sklearn)

    # ── Emissive tab builder ──────────────────────────────────────────────────

    def _build_emissive_tab(self):
        """Build all controls for the Emissive Dreamy tab."""
        d = EMISSIVE_DEFAULTS

        # Inner widget that holds all the collapsible groups
        inner = QWidget()
        inner_layout = QVBoxLayout(inner)
        inner_layout.setContentsMargins(4, 4, 4, 4)
        inner_layout.setSpacing(6)

        def _slider_row(label, slider, spin):
            row = QWidget()
            h = QHBoxLayout(row)
            h.setContentsMargins(0, 0, 0, 0)
            h.addWidget(QLabel(label))
            h.addWidget(slider, 1)
            h.addWidget(spin)
            return row

        # --- Preset selector ---
        self._em_preset_combo = QComboBox()
        self._em_preset_combo.addItems(["Default Emissive", "Custom"])
        self._em_preset_combo.currentIndexChanged.connect(self._on_emissive_preset_changed)
        inner_layout.addWidget(QLabel("Preset"))
        inner_layout.addWidget(self._em_preset_combo)

        # --- 1. Contrast group ---
        contrast_grp = CollapsibleGroupBox("Contrast")

        self._em_shadow_slider = QSlider(Qt.Horizontal); self._em_shadow_slider.setRange(0, 128); self._em_shadow_slider.setValue(d["levels_shadow"])
        self._em_shadow_spin = QSpinBox(); self._em_shadow_spin.setRange(0, 128); self._em_shadow_spin.setValue(d["levels_shadow"])
        self._em_shadow_slider.valueChanged.connect(self._em_shadow_spin.setValue)
        self._em_shadow_spin.valueChanged.connect(self._em_shadow_slider.setValue)
        self._em_shadow_spin.valueChanged.connect(self._mark_emissive_custom)
        self._em_shadow_spin.valueChanged.connect(self.schedule_preview)
        contrast_grp.addWidget(_slider_row("Shadow", self._em_shadow_slider, self._em_shadow_spin))

        self._em_highlight_slider = QSlider(Qt.Horizontal); self._em_highlight_slider.setRange(128, 255); self._em_highlight_slider.setValue(d["levels_highlight"])
        self._em_highlight_spin = QSpinBox(); self._em_highlight_spin.setRange(128, 255); self._em_highlight_spin.setValue(d["levels_highlight"])
        self._em_highlight_slider.valueChanged.connect(self._em_highlight_spin.setValue)
        self._em_highlight_spin.valueChanged.connect(self._em_highlight_slider.setValue)
        self._em_highlight_spin.valueChanged.connect(self._mark_emissive_custom)
        self._em_highlight_spin.valueChanged.connect(self.schedule_preview)
        contrast_grp.addWidget(_slider_row("Highlight", self._em_highlight_slider, self._em_highlight_spin))

        self._em_gamma_slider = QSlider(Qt.Horizontal); self._em_gamma_slider.setRange(10, 300); self._em_gamma_slider.setValue(int(d["levels_gamma"] * 100))
        self._em_gamma_spin = QDoubleSpinBox(); self._em_gamma_spin.setRange(0.1, 3.0); self._em_gamma_spin.setSingleStep(0.05); self._em_gamma_spin.setDecimals(2); self._em_gamma_spin.setValue(d["levels_gamma"])
        self._em_gamma_slider.valueChanged.connect(lambda v: self._em_gamma_spin.setValue(v / 100.0))
        self._em_gamma_spin.valueChanged.connect(lambda v: self._em_gamma_slider.setValue(int(v * 100)))
        self._em_gamma_spin.valueChanged.connect(self._mark_emissive_custom)
        self._em_gamma_spin.valueChanged.connect(self.schedule_preview)
        contrast_grp.addWidget(_slider_row("Gamma", self._em_gamma_slider, self._em_gamma_spin))

        inner_layout.addWidget(contrast_grp)

        # --- 2. Color Grading group ---
        color_grp = CollapsibleGroupBox("Color Grading")

        self._em_color_mode = QComboBox()
        self._em_color_mode.addItems(["Gradient Map", "Colorize"])
        self._em_color_mode.currentIndexChanged.connect(self._update_color_mode_visibility)
        self._em_color_mode.currentIndexChanged.connect(self._mark_emissive_custom)
        self._em_color_mode.currentIndexChanged.connect(self.schedule_preview)
        color_grp.addWidget(QLabel("Mode"))
        color_grp.addWidget(self._em_color_mode)

        # Gradient map colours
        self._em_gm_shadow_btn = ColorButton(d["gm_shadow"])
        self._em_gm_mid_btn = ColorButton(d["gm_mid"])
        self._em_gm_highlight_btn = ColorButton(d["gm_highlight"])
        for btn in (self._em_gm_shadow_btn, self._em_gm_mid_btn, self._em_gm_highlight_btn):
            btn.color_changed.connect(self._mark_emissive_custom)
            btn.color_changed.connect(self.schedule_preview)

        self._gm_widget = QWidget()
        gm_l = QVBoxLayout(self._gm_widget); gm_l.setContentsMargins(0, 0, 0, 0); gm_l.setSpacing(2)
        gm_l.addWidget(QLabel("Shadow")); gm_l.addWidget(self._em_gm_shadow_btn)
        gm_l.addWidget(QLabel("Midtone")); gm_l.addWidget(self._em_gm_mid_btn)
        gm_l.addWidget(QLabel("Highlight")); gm_l.addWidget(self._em_gm_highlight_btn)
        color_grp.addWidget(self._gm_widget)

        # Colorize controls
        self._em_hue_slider = QSlider(Qt.Horizontal); self._em_hue_slider.setRange(0, 360); self._em_hue_slider.setValue(d["colorize_hue"])
        self._em_hue_spin = QSpinBox(); self._em_hue_spin.setRange(0, 360); self._em_hue_spin.setValue(d["colorize_hue"]); self._em_hue_spin.setSuffix("°")
        self._em_hue_slider.valueChanged.connect(self._em_hue_spin.setValue)
        self._em_hue_spin.valueChanged.connect(self._em_hue_slider.setValue)
        self._em_hue_spin.valueChanged.connect(self._mark_emissive_custom)
        self._em_hue_spin.valueChanged.connect(self.schedule_preview)

        self._em_sat_slider = QSlider(Qt.Horizontal); self._em_sat_slider.setRange(0, 100); self._em_sat_slider.setValue(d["colorize_sat"])
        self._em_sat_spin = QSpinBox(); self._em_sat_spin.setRange(0, 100); self._em_sat_spin.setValue(d["colorize_sat"]); self._em_sat_spin.setSuffix(" %")
        self._em_sat_slider.valueChanged.connect(self._em_sat_spin.setValue)
        self._em_sat_spin.valueChanged.connect(self._em_sat_slider.setValue)
        self._em_sat_spin.valueChanged.connect(self._mark_emissive_custom)
        self._em_sat_spin.valueChanged.connect(self.schedule_preview)

        self._colorize_widget = QWidget()
        cl_l = QVBoxLayout(self._colorize_widget); cl_l.setContentsMargins(0, 0, 0, 0); cl_l.setSpacing(2)
        cl_l.addWidget(_slider_row("Hue", self._em_hue_slider, self._em_hue_spin))
        cl_l.addWidget(_slider_row("Sat", self._em_sat_slider, self._em_sat_spin))
        color_grp.addWidget(self._colorize_widget)

        self._update_color_mode_visibility()
        inner_layout.addWidget(color_grp)

        # --- 3. Glow group ---
        glow_grp = CollapsibleGroupBox("Emissive Glow")

        self._em_glow_check = QCheckBox("Enable"); self._em_glow_check.setChecked(d["glow_enabled"])
        self._em_glow_check.toggled.connect(self._mark_emissive_custom)
        self._em_glow_check.toggled.connect(self.schedule_preview)
        glow_grp.addWidget(self._em_glow_check)

        self._em_glow_blur_slider = QSlider(Qt.Horizontal); self._em_glow_blur_slider.setRange(1, 80); self._em_glow_blur_slider.setValue(int(d["glow_blur"]))
        self._em_glow_blur_spin = QSpinBox(); self._em_glow_blur_spin.setRange(1, 80); self._em_glow_blur_spin.setValue(int(d["glow_blur"]))
        self._em_glow_blur_slider.valueChanged.connect(self._em_glow_blur_spin.setValue)
        self._em_glow_blur_spin.valueChanged.connect(self._em_glow_blur_slider.setValue)
        self._em_glow_blur_spin.valueChanged.connect(self._mark_emissive_custom)
        self._em_glow_blur_spin.valueChanged.connect(self.schedule_preview)
        glow_grp.addWidget(_slider_row("Blur", self._em_glow_blur_slider, self._em_glow_blur_spin))

        self._em_glow_blend = QComboBox(); self._em_glow_blend.addItems(["Screen", "Linear Dodge", "Color Dodge"])
        self._em_glow_blend.currentIndexChanged.connect(self._mark_emissive_custom)
        self._em_glow_blend.currentIndexChanged.connect(self.schedule_preview)
        glow_grp.addWidget(QLabel("Blend")); glow_grp.addWidget(self._em_glow_blend)

        self._em_glow_opacity_slider = QSlider(Qt.Horizontal); self._em_glow_opacity_slider.setRange(0, 100); self._em_glow_opacity_slider.setValue(int(d["glow_opacity"] * 100))
        self._em_glow_opacity_spin = QSpinBox(); self._em_glow_opacity_spin.setRange(0, 100); self._em_glow_opacity_spin.setValue(int(d["glow_opacity"] * 100)); self._em_glow_opacity_spin.setSuffix(" %")
        self._em_glow_opacity_slider.valueChanged.connect(self._em_glow_opacity_spin.setValue)
        self._em_glow_opacity_spin.valueChanged.connect(self._em_glow_opacity_slider.setValue)
        self._em_glow_opacity_spin.valueChanged.connect(self._mark_emissive_custom)
        self._em_glow_opacity_spin.valueChanged.connect(self.schedule_preview)
        glow_grp.addWidget(_slider_row("Opacity", self._em_glow_opacity_slider, self._em_glow_opacity_spin))

        self._em_glow_thresh_slider = QSlider(Qt.Horizontal); self._em_glow_thresh_slider.setRange(0, 255); self._em_glow_thresh_slider.setValue(d["glow_threshold"])
        self._em_glow_thresh_spin = QSpinBox(); self._em_glow_thresh_spin.setRange(0, 255); self._em_glow_thresh_spin.setValue(d["glow_threshold"])
        self._em_glow_thresh_slider.valueChanged.connect(self._em_glow_thresh_spin.setValue)
        self._em_glow_thresh_spin.valueChanged.connect(self._em_glow_thresh_slider.setValue)
        self._em_glow_thresh_spin.valueChanged.connect(self._mark_emissive_custom)
        self._em_glow_thresh_spin.valueChanged.connect(self.schedule_preview)
        glow_grp.addWidget(_slider_row("Threshold", self._em_glow_thresh_slider, self._em_glow_thresh_spin))

        inner_layout.addWidget(glow_grp)

        # --- 4. Softness group ---
        soft_grp = CollapsibleGroupBox("Dream Softness")

        self._em_soft_check = QCheckBox("Enable"); self._em_soft_check.setChecked(d["soft_enabled"])
        self._em_soft_check.toggled.connect(self._mark_emissive_custom)
        self._em_soft_check.toggled.connect(self.schedule_preview)
        soft_grp.addWidget(self._em_soft_check)

        self._em_soft_blur_slider = QSlider(Qt.Horizontal); self._em_soft_blur_slider.setRange(5, 120); self._em_soft_blur_slider.setValue(int(d["soft_blur"] * 10))
        self._em_soft_blur_spin = QDoubleSpinBox(); self._em_soft_blur_spin.setRange(0.5, 12.0); self._em_soft_blur_spin.setSingleStep(0.5); self._em_soft_blur_spin.setDecimals(1); self._em_soft_blur_spin.setValue(d["soft_blur"])
        self._em_soft_blur_slider.valueChanged.connect(lambda v: self._em_soft_blur_spin.setValue(v / 10.0))
        self._em_soft_blur_spin.valueChanged.connect(lambda v: self._em_soft_blur_slider.setValue(int(v * 10)))
        self._em_soft_blur_spin.valueChanged.connect(self._mark_emissive_custom)
        self._em_soft_blur_spin.valueChanged.connect(self.schedule_preview)
        soft_grp.addWidget(_slider_row("Blur", self._em_soft_blur_slider, self._em_soft_blur_spin))

        self._em_soft_blend = QComboBox(); self._em_soft_blend.addItems(["Soft Light", "Overlay"])
        self._em_soft_blend.currentIndexChanged.connect(self._mark_emissive_custom)
        self._em_soft_blend.currentIndexChanged.connect(self.schedule_preview)
        soft_grp.addWidget(QLabel("Blend")); soft_grp.addWidget(self._em_soft_blend)

        self._em_soft_opacity_slider = QSlider(Qt.Horizontal); self._em_soft_opacity_slider.setRange(0, 50); self._em_soft_opacity_slider.setValue(int(d["soft_opacity"] * 100))
        self._em_soft_opacity_spin = QSpinBox(); self._em_soft_opacity_spin.setRange(0, 50); self._em_soft_opacity_spin.setValue(int(d["soft_opacity"] * 100)); self._em_soft_opacity_spin.setSuffix(" %")
        self._em_soft_opacity_slider.valueChanged.connect(self._em_soft_opacity_spin.setValue)
        self._em_soft_opacity_spin.valueChanged.connect(self._em_soft_opacity_slider.setValue)
        self._em_soft_opacity_spin.valueChanged.connect(self._mark_emissive_custom)
        self._em_soft_opacity_spin.valueChanged.connect(self.schedule_preview)
        soft_grp.addWidget(_slider_row("Opacity", self._em_soft_opacity_slider, self._em_soft_opacity_spin))

        inner_layout.addWidget(soft_grp)

        # --- 5. Color Bloom group ---
        bloom_grp = CollapsibleGroupBox("Color Bloom", collapsed=True)

        self._em_bloom_check = QCheckBox("Enable"); self._em_bloom_check.setChecked(d["bloom_enabled"])
        self._em_bloom_check.toggled.connect(self._mark_emissive_custom)
        self._em_bloom_check.toggled.connect(self.schedule_preview)
        bloom_grp.addWidget(self._em_bloom_check)

        self._em_bloom_color_btn = ColorButton(d["bloom_color"])
        self._em_bloom_color_btn.color_changed.connect(self._mark_emissive_custom)
        self._em_bloom_color_btn.color_changed.connect(self.schedule_preview)
        bloom_grp.addWidget(QLabel("Color")); bloom_grp.addWidget(self._em_bloom_color_btn)

        self._em_bloom_blend = QComboBox(); self._em_bloom_blend.addItems(["Overlay", "Soft Light"])
        self._em_bloom_blend.currentIndexChanged.connect(self._mark_emissive_custom)
        self._em_bloom_blend.currentIndexChanged.connect(self.schedule_preview)
        bloom_grp.addWidget(QLabel("Blend")); bloom_grp.addWidget(self._em_bloom_blend)

        self._em_bloom_opacity_slider = QSlider(Qt.Horizontal); self._em_bloom_opacity_slider.setRange(0, 50); self._em_bloom_opacity_slider.setValue(int(d["bloom_opacity"] * 100))
        self._em_bloom_opacity_spin = QSpinBox(); self._em_bloom_opacity_spin.setRange(0, 50); self._em_bloom_opacity_spin.setValue(int(d["bloom_opacity"] * 100)); self._em_bloom_opacity_spin.setSuffix(" %")
        self._em_bloom_opacity_slider.valueChanged.connect(self._em_bloom_opacity_spin.setValue)
        self._em_bloom_opacity_spin.valueChanged.connect(self._em_bloom_opacity_slider.setValue)
        self._em_bloom_opacity_spin.valueChanged.connect(self._mark_emissive_custom)
        self._em_bloom_opacity_spin.valueChanged.connect(self.schedule_preview)
        bloom_grp.addWidget(_slider_row("Opacity", self._em_bloom_opacity_slider, self._em_bloom_opacity_spin))

        inner_layout.addWidget(bloom_grp)

        # --- 6. Grain group ---
        grain_grp = CollapsibleGroupBox("Grain")

        self._em_grain_check = QCheckBox("Enable"); self._em_grain_check.setChecked(d["grain_enabled"])
        self._em_grain_check.toggled.connect(self._mark_emissive_custom)
        self._em_grain_check.toggled.connect(self.schedule_preview)
        grain_grp.addWidget(self._em_grain_check)

        self._em_grain_slider = QSlider(Qt.Horizontal); self._em_grain_slider.setRange(0, 100); self._em_grain_slider.setValue(int(d["grain_intensity"] * 100))
        self._em_grain_spin = QSpinBox(); self._em_grain_spin.setRange(0, 100); self._em_grain_spin.setValue(int(d["grain_intensity"] * 100)); self._em_grain_spin.setSuffix(" %")
        self._em_grain_slider.valueChanged.connect(self._em_grain_spin.setValue)
        self._em_grain_spin.valueChanged.connect(self._em_grain_slider.setValue)
        self._em_grain_spin.valueChanged.connect(self._mark_emissive_custom)
        self._em_grain_spin.valueChanged.connect(self.schedule_preview)
        grain_grp.addWidget(_slider_row("Intensity", self._em_grain_slider, self._em_grain_spin))

        self._em_grain_type = QComboBox(); self._em_grain_type.addItems(["Gaussian", "Uniform"])
        self._em_grain_type.currentIndexChanged.connect(self._mark_emissive_custom)
        self._em_grain_type.currentIndexChanged.connect(self.schedule_preview)
        grain_grp.addWidget(QLabel("Type")); grain_grp.addWidget(self._em_grain_type)

        inner_layout.addWidget(grain_grp)
        inner_layout.addStretch()

        # Wrap in scroll area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(inner)
        scroll.setFrameShape(QFrame.NoFrame)

        self._right_tabs.addTab(scroll, "Emissive Dreamy")

    # ── Emissive helpers ──────────────────────────────────────────────────────

    def _update_color_mode_visibility(self, _idx=None):
        is_gm = self._em_color_mode.currentIndex() == 0
        self._gm_widget.setVisible(is_gm)
        self._colorize_widget.setVisible(not is_gm)

    def _mark_emissive_custom(self, *_args):
        """Switch preset dropdown to 'Custom' when user tweaks any control."""
        if self._em_preset_combo.currentIndex() != 1:
            self._em_preset_combo.blockSignals(True)
            self._em_preset_combo.setCurrentIndex(1)
            self._em_preset_combo.blockSignals(False)

    def _on_emissive_preset_changed(self, idx):
        if idx == 0:
            self._apply_emissive_defaults()

    def _apply_emissive_defaults(self):
        """Reset all emissive controls to EMISSIVE_DEFAULTS."""
        d = EMISSIVE_DEFAULTS
        # Block signals to avoid N preview triggers — we trigger once at the end
        widgets = [
            self._em_shadow_spin, self._em_highlight_spin, self._em_gamma_spin,
            self._em_color_mode, self._em_hue_spin, self._em_sat_spin,
            self._em_glow_check, self._em_glow_blur_spin, self._em_glow_blend,
            self._em_glow_opacity_spin, self._em_glow_thresh_spin,
            self._em_soft_check, self._em_soft_blur_spin, self._em_soft_blend,
            self._em_soft_opacity_spin,
            self._em_bloom_check, self._em_bloom_blend, self._em_bloom_opacity_spin,
            self._em_grain_check, self._em_grain_spin, self._em_grain_type,
        ]
        for w in widgets:
            w.blockSignals(True)

        self._em_shadow_spin.setValue(d["levels_shadow"])
        self._em_highlight_spin.setValue(d["levels_highlight"])
        self._em_gamma_spin.setValue(d["levels_gamma"])
        self._em_color_mode.setCurrentIndex(0 if d["color_mode"] == "gradient_map" else 1)
        self._em_gm_shadow_btn.set_color(d["gm_shadow"])
        self._em_gm_mid_btn.set_color(d["gm_mid"])
        self._em_gm_highlight_btn.set_color(d["gm_highlight"])
        self._em_hue_spin.setValue(d["colorize_hue"])
        self._em_sat_spin.setValue(d["colorize_sat"])
        self._em_glow_check.setChecked(d["glow_enabled"])
        self._em_glow_blur_spin.setValue(int(d["glow_blur"]))
        self._em_glow_blend.setCurrentText(d["glow_blend"])
        self._em_glow_opacity_spin.setValue(int(d["glow_opacity"] * 100))
        self._em_glow_thresh_spin.setValue(d["glow_threshold"])
        self._em_soft_check.setChecked(d["soft_enabled"])
        self._em_soft_blur_spin.setValue(d["soft_blur"])
        self._em_soft_blend.setCurrentText(d["soft_blend"])
        self._em_soft_opacity_spin.setValue(int(d["soft_opacity"] * 100))
        self._em_bloom_check.setChecked(d["bloom_enabled"])
        self._em_bloom_color_btn.set_color(d["bloom_color"])
        self._em_bloom_blend.setCurrentText(d["bloom_blend"])
        self._em_bloom_opacity_spin.setValue(int(d["bloom_opacity"] * 100))
        self._em_grain_check.setChecked(d["grain_enabled"])
        self._em_grain_spin.setValue(int(d["grain_intensity"] * 100))
        self._em_grain_type.setCurrentText(d["grain_type"].capitalize())

        for w in widgets:
            w.blockSignals(False)

        self._update_color_mode_visibility()
        self.schedule_preview()

    def _collect_emissive_params(self) -> dict:
        """Read all emissive controls into a params dict for process_emissive()."""
        return {
            "levels_shadow": self._em_shadow_spin.value(),
            "levels_highlight": self._em_highlight_spin.value(),
            "levels_gamma": self._em_gamma_spin.value(),
            "color_mode": "gradient_map" if self._em_color_mode.currentIndex() == 0 else "colorize",
            "gm_shadow": self._em_gm_shadow_btn.color(),
            "gm_mid": self._em_gm_mid_btn.color(),
            "gm_highlight": self._em_gm_highlight_btn.color(),
            "colorize_hue": self._em_hue_spin.value(),
            "colorize_sat": self._em_sat_spin.value(),
            "glow_enabled": self._em_glow_check.isChecked(),
            "glow_blur": float(self._em_glow_blur_spin.value()),
            "glow_blend": self._em_glow_blend.currentText(),
            "glow_opacity": self._em_glow_opacity_spin.value() / 100.0,
            "glow_threshold": self._em_glow_thresh_spin.value(),
            "soft_enabled": self._em_soft_check.isChecked(),
            "soft_blur": self._em_soft_blur_spin.value(),
            "soft_blend": self._em_soft_blend.currentText(),
            "soft_opacity": self._em_soft_opacity_spin.value() / 100.0,
            "bloom_enabled": self._em_bloom_check.isChecked(),
            "bloom_color": self._em_bloom_color_btn.color(),
            "bloom_blend": self._em_bloom_blend.currentText(),
            "bloom_opacity": self._em_bloom_opacity_spin.value() / 100.0,
            "grain_enabled": self._em_grain_check.isChecked(),
            "grain_intensity": self._em_grain_spin.value() / 100.0,
            "grain_type": self._em_grain_type.currentText().lower(),
        }

    # ── Preview scheduling ────────────────────────────────────────────────────

    def schedule_preview(self, *_args):
        self._preview_timer.start()

    def _do_preview(self):
        if self._working_image is None:
            return
        self.apply_processing()

    # ── Image I/O ─────────────────────────────────────────────────────────────

    def open_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open image")
        if not path:
            return
        try:
            self.image = Image.open(path)
        except Exception:
            self.image = None
            return
        self._working_image = self.image.convert("RGBA").copy()
        self._preview_image = None
        self._undo_stack.clear()
        self._redo_stack.clear()
        self._apply_btn.setEnabled(True)
        self._update_undo_redo_actions()
        pix = QPixmap.fromImage(ImageQt(self._working_image))
        self._current_pixmap = pix
        self.preview.set_pixmap(pix, ref_size=self._working_image.size)  # resets zoom/pan

    def show_pil(self, pil_img: Image.Image):
        """Display a PIL image, preserving current zoom and pan."""
        qim = ImageQt(pil_img.convert("RGBA"))
        pix = QPixmap.fromImage(qim)
        self._current_pixmap = pix
        self.preview.set_pixmap_keep_view(pix)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        # canvas handles its own resize; nothing extra needed here

    # ── Zoom ──────────────────────────────────────────────────────────────────

    def eventFilter(self, obj, event):
        """Ctrl+scroll on the canvas → zoom anchored to cursor."""
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
        """Show the committed working image when switching tabs — never re-processes."""
        if self._working_image is None:
            return
        self._preview_image = None
        self.show_pil(self._working_image)

    def _show_after(self):
        self._showing_original = False
        img = self._preview_image if self._preview_image is not None else self._working_image
        if img is not None:
            self.show_pil(img)

    # ── Processing ────────────────────────────────────────────────────────────

    def _is_emissive_mode(self) -> bool:
        return self._right_tabs.currentIndex() == 1

    def _build_quant_inputs(self) -> tuple:
        """Read quantization tab controls. Returns (img, params_dict)."""
        scale = self.scale_spin.value() / 100.0
        blur = self.blur_spin.value()
        colors = self.colors_spin.value()
        dither = self.dither_check.isChecked()
        use_pillow_dither = self.dither_method_combo.currentIndex() == 1
        resample_idx = self.resample_combo.currentIndex()
        resample_filter = self.resample_map.get(resample_idx, Image.LANCZOS)
        img = self._working_image.copy()
        if scale != 1.0:
            w, h = img.size
            new_size = (max(1, int(w * scale)), max(1, int(h * scale)))
            img = img.resize(new_size, resample=resample_filter)
        img = img if blur <= 0 else img.filter(ImageFilter.GaussianBlur(blur))
        return img, {"colors": colors, "dither": dither, "use_pillow_dither": use_pillow_dither}

    def apply_processing(self):
        """Run a live draft preview on top of the current working image."""
        if self._working_image is None:
            return

        if self._is_emissive_mode():
            self._start_worker_generic(
                "emissive", self._working_image.copy(),
                self._collect_emissive_params(), draft=True, commit=False,
            )
        else:
            img, params = self._build_quant_inputs()
            self._start_worker_generic("quantize", img, params, draft=False, commit=False)

    def _apply_working(self):
        """Commit current settings onto the working image at full resolution."""
        if self._working_image is None:
            return
        self._apply_btn.setEnabled(False)
        self.statusBar().showMessage("Applying…")
        if self._is_emissive_mode():
            self._start_worker_generic(
                "emissive", self._working_image.copy(),
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
        self.statusBar().showMessage(
            f"Undo — {len(self._undo_stack)} step(s) remaining", 2000)

    def _redo(self):
        if not self._redo_stack:
            return
        self._undo_stack.append(self._working_image)
        self._working_image = self._redo_stack.pop()
        self._preview_image = None
        self.show_pil(self._working_image)
        self._update_undo_redo_actions()
        self.statusBar().showMessage(
            f"Redo — {len(self._undo_stack)} step(s) in history", 2000)

    def _update_undo_redo_actions(self):
        self._undo_action.setEnabled(bool(self._undo_stack))
        self._redo_action.setEnabled(bool(self._redo_stack))

    def _start_worker_generic(self, mode, img, params, draft=False, commit=False):
        """Unified worker submission for any processing pipeline."""
        if self._current_future is not None and not self._current_future.done():
            self._pending_params = (mode, img, params, draft, commit)
            return

        self._current_job_commit = commit

        def _worker(mode_, img_, params_, draft_):
            if mode_ == "emissive":
                return process_emissive(img_, params_, draft=draft_)
            else:
                return process_image(
                    img_, 1.0, 0.0,
                    params_["colors"], params_["dither"],
                    use_pillow_dither=params_["use_pillow_dither"],
                )

        if not commit:
            self.statusBar().showMessage("Preview…")
        future = self._process_pool.submit(_worker, mode, img, params, draft)
        self._current_future = future
        future.add_done_callback(lambda f: self.processing_done.emit(f))

    def save_result(self):
        """Save the committed working image directly — it is always full resolution."""
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

        if data:
            try:
                if isinstance(data, Image.Image):
                    if self._current_job_commit:
                        # Commit: push old working image to undo stack, store new one
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


def main():
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())
