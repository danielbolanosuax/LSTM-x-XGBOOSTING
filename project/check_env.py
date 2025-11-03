# path: project/check_env.py
from __future__ import annotations
import sys
print("Python:", sys.version)
try:
    import tensorflow as tf
    print("TensorFlow:", tf.__version__)
except Exception as e:
    print("TensorFlow: NOT FOUND ->", e)
try:
    from project.nn_compat import backend_name
    print("Keras backend:", backend_name())
except Exception as e:
    print("nn_compat error:", e)
