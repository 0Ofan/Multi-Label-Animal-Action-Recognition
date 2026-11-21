"""Streamlit Community Cloud and local single-process entry point."""
import os,runpy
from pathlib import Path
os.environ.setdefault("ANIMALACT2_BACKEND","direct")
runpy.run_path(str(Path(__file__).resolve().parent/"animalact2/dashboard/ui/app.py"),run_name="__main__")
