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
    QMenuBar,
    QGroupBox,
    QSizePolicy,
    QComboBox,
)
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QPixmap, QAction
from PIL.ImageQt import ImageQt
from PIL import Image, ImageFilter
import sys
import traceback

from .processing import process_image

import concurrent.futures
import threading


class MainWindow(QMainWindow):
    processing_done = Signal(object)
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Python prototype")
        self.image = None
        self.result = None
        # Background worker state (thread pool)
        self._process_pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        self._current_future = None
        self._pending_params = None

        # connect processing_done signal to handler so callbacks run on GUI thread
        self.processing_done.connect(self._on_future_done)

        # Menu bar (top strip for menus only)
        menubar = QMenuBar(self)
        file_menu = menubar.addMenu("File")

        open_action = QAction("Open", self)
        save_action = QAction("Save", self)
        exit_action = QAction("Exit", self)

        file_menu.addAction(open_action)
        file_menu.addAction(save_action)
        file_menu.addSeparator()
        file_menu.addAction(exit_action)

        open_action.triggered.connect(self.open_image)
        save_action.triggered.connect(self.save_result)
        exit_action.triggered.connect(self.close)

        self.setMenuBar(menubar)

        # Central layout: left tools | preview | right tools
        central = QWidget()
        central_layout = QHBoxLayout()
        central.setLayout(central_layout)
        self.setCentralWidget(central)

        # Left group (tools/actions)
        left_group = QGroupBox("Actions")
        left_layout = QVBoxLayout()
        left_group.setLayout(left_layout)

        self.open_btn = QPushButton("Open Image")
        self.save_btn = QPushButton("Save Result")
        self.apply_btn = QPushButton("Apply")

        left_layout.addWidget(self.open_btn)
        left_layout.addWidget(self.save_btn)
        left_layout.addStretch()
        left_layout.addWidget(self.apply_btn)

        # Right group (processing options)
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

        self.dither_check = QCheckBox("Dither (Floyd–Steinberg)")
        self.dither_check.setChecked(True)

        self.resample_combo = QComboBox()
        self.resample_combo.addItems(["Nearest", "Bilinear", "Bicubic", "Lanczos"])
        self.resample_map = {
            0: Image.NEAREST,
            1: Image.BILINEAR,
            2: Image.BICUBIC,
            3: Image.LANCZOS,
        }

        right_layout.addWidget(QLabel("Scale"))
        scale_container = QWidget()
        scale_h = QHBoxLayout()
        scale_h.setContentsMargins(0, 0, 0, 0)
        scale_container.setLayout(scale_h)
        scale_h.addWidget(self.scale_slider)
        scale_h.addWidget(self.scale_spin)
        right_layout.addWidget(scale_container)

        right_layout.addWidget(QLabel("Blur"))
        blur_container = QWidget()
        blur_h = QHBoxLayout()
        blur_h.setContentsMargins(0, 0, 0, 0)
        blur_container.setLayout(blur_h)
        blur_h.addWidget(self.blur_slider)
        blur_h.addWidget(self.blur_spin)
        right_layout.addWidget(blur_container)
        right_layout.addWidget(QLabel("Colors"))
        right_layout.addWidget(self.colors_spin)
        right_layout.addWidget(self.dither_check)
        right_layout.addWidget(QLabel("Resample"))
        right_layout.addWidget(self.resample_combo)
        right_layout.addStretch()

        # Preview area
        preview_container = QWidget()
        preview_layout = QVBoxLayout()
        preview_container.setLayout(preview_layout)

        self.preview = QLabel(alignment=Qt.AlignCenter)
        self.preview.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.preview.setMinimumSize(200, 200)
        self.preview.setStyleSheet("background-color: #222; color: #ddd;")
        self.preview.setText("No image loaded")

        preview_layout.addWidget(self.preview)

        central_layout.addWidget(left_group)
        central_layout.addWidget(preview_container, 1)
        central_layout.addWidget(right_group)

        # Connections
        self.open_btn.clicked.connect(self.open_image)
        self.save_btn.clicked.connect(self.save_result)
        self.apply_btn.clicked.connect(self.apply_processing)

        # Live preview debounce timer
        self._preview_timer = QTimer(self)
        self._preview_timer.setSingleShot(True)
        self._preview_timer.setInterval(300)
        self._preview_timer.timeout.connect(self._do_preview)

        # Wire control changes to schedule preview and sync widgets
        self.scale_slider.valueChanged.connect(self.scale_spin.setValue)
        self.scale_spin.valueChanged.connect(self.scale_slider.setValue)
        self.scale_spin.valueChanged.connect(self.schedule_preview)

        # Blur: slider uses integer half-steps (0..50) while spinbox shows actual radius (0.0..25.0)
        self.blur_slider.valueChanged.connect(lambda v: self.blur_spin.setValue(v / 2.0))
        self.blur_spin.valueChanged.connect(lambda v: self.blur_slider.setValue(int(v * 2)))
        self.blur_spin.valueChanged.connect(self.schedule_preview)

        self.colors_spin.valueChanged.connect(self.schedule_preview)
        self.dither_check.toggled.connect(self.schedule_preview)
        self.resample_combo.currentIndexChanged.connect(self.schedule_preview)

        self.resize(1000, 700)

        # ensure there is a status bar for simple busy feedback
        self.statusBar().showMessage("")

    def schedule_preview(self, *_args):
        # start or restart debounce timer
        self._preview_timer.start()

    def _do_preview(self):
        # reuse apply_processing but don't block if no image
        if self.image is None:
            return
        self.apply_processing()

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
        self.show_pil(self.image)

    def show_pil(self, pil_img: Image.Image):
        qim = ImageQt(pil_img.convert("RGBA"))
        pix = QPixmap.fromImage(qim)
        target = self.preview.size()
        if target.width() <= 0 or target.height() <= 0:
            self.preview.setPixmap(pix)
            return
        scaled = pix.scaled(target, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.preview.setPixmap(scaled)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.result is not None:
            self.show_pil(self.result)
        elif self.image is not None:
            self.show_pil(self.image)

    def apply_processing(self):
        if self.image is None:
            return
        scale = self.scale_spin.value() / 100.0
        blur = self.blur_spin.value()
        colors = self.colors_spin.value()
        dither = self.dither_check.isChecked()
        resample_idx = self.resample_combo.currentIndex()
        resample_filter = self.resample_map.get(resample_idx, Image.LANCZOS)

        # Apply resample by resizing with chosen filter inside processing pipeline
        img = self.image.copy()
        # ensure the image data is loaded from disk into memory (avoid lazy file handles)
        try:
            img.load()
        except Exception:
            pass
        if scale != 1.0:
            w, h = img.size
            new_size = (max(1, int(w * scale)), max(1, int(h * scale)))
            img = img.resize(new_size, resample=resample_filter)

        img = img if blur <= 0 else img.filter(ImageFilter.GaussianBlur(blur))

        # Use background worker to avoid blocking the UI
        self._start_worker(img, colors, dither)

    def _start_worker(self, img, colors, dither):
        # If a job is already running, store the latest params and return (coalesce)
        if self._current_future is not None and not self._current_future.done():
            self._pending_params = (img, colors, dither)
            return

        print("[gui] Submitting processing job: colors=", colors, "dither=", dither)
        sys.stdout.flush()

        # wrapper around process_image to trace start/finish inside the worker thread
        def _process_wrapper(img_, scale_, blur_, colors_, dither_):
            print(f"[worker:{threading.get_ident()}] started process_image: colors={colors_} dither={dither_}")
            sys.stdout.flush()
            res = process_image(img_, scale_, blur_, colors_, dither_)
            print(f"[worker:{threading.get_ident()}] finished process_image")
            sys.stdout.flush()
            return res

        # Submit processing to thread pool; pass PIL Image directly
        future = self._process_pool.submit(_process_wrapper, img, 1.0, 0.0, colors, dither)
        self._current_future = future

        # simple UI feedback
        self.statusBar().showMessage("Processing...")
        self.apply_btn.setEnabled(False)

        # Use a Qt signal to marshal the Future to the GUI thread
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

        # clear busy state
        self.statusBar().clearMessage()
        self.apply_btn.setEnabled(True)

        # if there was a newer request while we were running, start it now
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
            # show short message to user
            try:
                self.statusBar().showMessage(f"Processing failed: {exc}")
            except Exception:
                pass
            data = None

        if data:
            try:
                # `data` is a PIL Image returned from the thread pool
                if isinstance(data, Image.Image):
                    self.result = data
                else:
                    # unexpected type: try to load as bytes
                    try:
                        buf = io.BytesIO(data)
                        img = Image.open(buf).copy()
                        self.result = img
                    except Exception:
                        self.result = None

                if self.result is not None:
                    self.show_pil(self.result)
            except Exception as exc:
                print("Failed to set result image:", exc)

        # clear busy state
        print("[gui] Clearing busy state")
        self.statusBar().clearMessage()
        self.apply_btn.setEnabled(True)

        # clear current future
        self._current_future = None

        # handle queued params
        if self._pending_params is not None:
            params = self._pending_params
            self._pending_params = None
            self._start_worker(*params)

    def closeEvent(self, event):
        # Attempt to stop any running worker thread promptly
        # If a process-based worker is running, try to cancel and shutdown pool
        try:
            if self._current_future is not None and not self._current_future.done():
                try:
                    self.statusBar().showMessage("Cancelling background processing...")
                    self._current_future.cancel()
                except Exception:
                    pass

            # attempt to shutdown pool without waiting
            try:
                self._process_pool.shutdown(wait=False)
            except Exception:
                pass

            # try to terminate any leftover worker processes (best-effort)
            try:
                procs = getattr(self._process_pool, "_processes", {})
                for p in list(procs.values()):
                    try:
                        p.terminate()
                        p.join(1)
                    except Exception:
                        pass
            except Exception:
                pass
        except Exception:
            pass
        super().closeEvent(event)

    def save_result(self):
        # Save current result or image if no processed result
        target = self.result or self.image
        if target is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save result", filter="PNG Files (*.png);;All Files (*)")
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
