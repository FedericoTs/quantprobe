"""`python -m quantprobe` — identical to the `quantprobe` console script.

The PATH-proof entry point: on Windows, `pip install` often lands the .exe in a
user-site Scripts folder that is not on PATH; this always works regardless.
"""
from .cli import main

if __name__ == "__main__":
    main()
