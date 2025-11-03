# path: project/nn_compat.py
from __future__ import annotations

_BACKEND = None

try:
    import tensorflow as tf  # noqa: F401
    from tensorflow.keras import models as _models
    from tensorflow.keras import layers as _layers
    from tensorflow.keras import optimizers as _optimizers
    from tensorflow.keras import callbacks as _callbacks
    _BACKEND = "tf.keras"
except Exception:
    try:
        import keras  # noqa: F401
        from keras import models as _models
        from keras import layers as _layers
        from keras import optimizers as _optimizers
        from keras import callbacks as _callbacks
        _BACKEND = "keras"
    except Exception as e:
        raise ImportError(
            "No hay backend Keras.\n"
            "Instala TensorFlow 2.x:  python -m pip install tensorflow==2.20.*\n"
            "o Keras 3 (con backend tensorflow)."
        ) from e

Sequential = _models.Sequential
LSTM = _layers.LSTM
Dense = _layers.Dense
Dropout = _layers.Dropout
Adam = _optimizers.Adam
EarlyStopping = _callbacks.EarlyStopping

def backend_name() -> str:
    return _BACKEND
