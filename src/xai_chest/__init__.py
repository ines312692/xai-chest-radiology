"""Multimodal explainable AI for chest radiology.

This package is the library form of the Kaggle notebooks of the project. The notebooks
stay the record of what was run and what it produced. The package is what the thesis
code should be read from and re run: every step is a function with a single
responsibility, every experiment is a configuration file, and every entry point is a
script under scripts.

Reading order for someone discovering the project:
  1. config.py         what an experiment is made of
  2. data/splits.py    how the dataset becomes a leak free train validation test table
  3. training/engine.py how a backbone is fine tuned and selected
  4. explain/cam.py    how a saliency map is produced
  5. explain/evaluation.py how a saliency map is scored, with and without annotations
"""

__version__ = "1.0.0"
__all__ = ["config", "data", "models", "training", "explain", "reporting", "utils"]
