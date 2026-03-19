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
)
from PySide6.QtCore import Qt, QTimer, Signal, QSize, QPointF, QRectF
from PySide6.QtGui import QPixmap, QAction, QWheelEvent, QPainter, QCursor, QPalette
from PIL.ImageQt import ImageQt
from PIL import Image, ImageFilter
import sys
import traceback

from .processing import process_image, sklearn_available

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
# Main window
# ---------------------------------------------------------------------------

class MainWindow(QMainWindow):
    processing_done = Signal(object)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Python prototype")
        self.image = None
        self.result = None
        self._showing_original = False

        # Background worker state
        self._process_pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        self._current_future = None
        self._pending_params = None
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
        edit_menu.addAction(QAction("Undo\tCtrl+Z", self))
        edit_menu.addAction(QAction("Redo\tCtrl+Y", self))
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

        # Right panel
        right_group = QGroupBox("Processing")
        right_layout = QVBoxLayout()
        right_group.setLayout(right_layout)

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

        # Canvas
        self.preview = _PreviewCanvas()
        self.preview.installEventFilter(self)   # for Ctrl+scroll

        central_layout.addWidget(left_group)
        central_layout.addWidget(self.preview, 1)
        central_layout.addWidget(right_group)

        _PANEL_WIDTH = 250
        left_group.setFixedWidth(_PANEL_WIDTH)
        right_group.setFixedWidth(_PANEL_WIDTH)

        # ── Bottom toolbar ────────────────────────────────────────────────────
        bottom_toolbar = QToolBar("Tools", self)
        bottom_toolbar.setMovable(False)
        bottom_toolbar.setFloatable(False)
        bottom_toolbar.setIconSize(QSize(16, 16))
        self.addToolBar(Qt.BottomToolBarArea, bottom_toolbar)

        self._before_btn = QToolButton()
        self._before_btn.setText("⬛ Before / After")
        self._before_btn.setToolTip("Hold to preview the original image")
        self._before_btn.pressed.connect(self._show_before)
        self._before_btn.released.connect(self._show_after)
        bottom_toolbar.addWidget(self._before_btn)
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

    # ── Preview scheduling ────────────────────────────────────────────────────

    def schedule_preview(self, *_args):
        self._preview_timer.start()

    def _do_preview(self):
        if self.image is None:
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
        self.result = None
        pix = QPixmap.fromImage(ImageQt(self.image.convert("RGBA")))
        self._current_pixmap = pix
        self.preview.set_pixmap(pix, ref_size=self.image.size)  # resets zoom/pan

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

    def _show_after(self):
        self._showing_original = False
        img = self.result if self.result is not None else self.image
        if img is not None:
            self.show_pil(img)

    # ── Processing ────────────────────────────────────────────────────────────

    def apply_processing(self):
        if self.image is None:
            return
        scale = self.scale_spin.value() / 100.0
        blur = self.blur_spin.value()
        colors = self.colors_spin.value()
        dither = self.dither_check.isChecked()
        use_pillow_dither = self.dither_method_combo.currentIndex() == 1
        resample_idx = self.resample_combo.currentIndex()
        resample_filter = self.resample_map.get(resample_idx, Image.LANCZOS)

        img = self.image.copy()
        try:
            img.load()
        except Exception:
            pass
        if scale != 1.0:
            w, h = img.size
            new_size = (max(1, int(w * scale)), max(1, int(h * scale)))
            img = img.resize(new_size, resample=resample_filter)
        img = img if blur <= 0 else img.filter(ImageFilter.GaussianBlur(blur))
        self._start_worker(img, colors, dither, use_pillow_dither)

    def _start_worker(self, img, colors, dither, use_pillow_dither=False):
        if self._current_future is not None and not self._current_future.done():
            self._pending_params = (img, colors, dither, use_pillow_dither)
            return

        print("[gui] Submitting processing job: colors=", colors, "dither=", dither, "pillow=", use_pillow_dither)
        sys.stdout.flush()

        def _process_wrapper(img_, scale_, blur_, colors_, dither_, use_pillow_dither_):
            print(f"[worker:{threading.get_ident()}] started process_image")
            sys.stdout.flush()
            res = process_image(img_, scale_, blur_, colors_, dither_, use_pillow_dither=use_pillow_dither_)
            print(f"[worker:{threading.get_ident()}] finished process_image")
            sys.stdout.flush()
            return res

        future = self._process_pool.submit(_process_wrapper, img, 1.0, 0.0, colors, dither, use_pillow_dither)
        self._current_future = future
        self.statusBar().showMessage("Processing...")
        future.add_done_callback(lambda f: self.processing_done.emit(f))
        print("[gui] future running=", future.running(), "done=", future.done())
        sys.stdout.flush()

    def _on_worker_finished(self, result):
        if result is not None:
            self.result = result
            try:
                self.show_pil(self.result)
            except Exception:
                pass
        self.statusBar().clearMessage()
        if self._pending_params is not None:
            params = self._pending_params
            self._pending_params = None
            self._start_worker(*params)

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
                    self.result = data
                else:
                    self.result = None
                if self.result is not None:
                    self.show_pil(self.result)
            except Exception as exc:
                print("Failed to set result image:", exc)

        print("[gui] Clearing busy state")
        self.statusBar().clearMessage()
        self._current_future = None
        self._update_dither_controls()

        if self._pending_params is not None:
            params = self._pending_params
            self._pending_params = None
            self._start_worker(*params)

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

    def save_result(self):
        target = self.result or self.image
        if target is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save result", filter="PNG Files (*.png);;All Files (*)")
        if not path:
            return
        try:
            target.save(path)
        except Exception:
            pass


def main():
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())
