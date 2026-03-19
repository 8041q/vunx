# app package

__version__ = "0.1.0"

def run():
    # Import GUI only when run() is called to avoid importing GUI code
    # inside child processes when using multiprocessing on Windows.
    from .gui import main as _gui_main
    _gui_main()
