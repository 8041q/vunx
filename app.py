# Top-level runner to preserve package context for Nuitka builds.
# run at root `python -m nuitka app`

from src import run

if __name__ == "__main__":
    run()
