try:
    import cupy as np
    GPU_AVAILABLE = True
except ImportError:
    import numpy as np
    GPU_AVAILABLE = False


def to_numpy(x):
    if hasattr(x, 'value'):
        x = x.value
    if GPU_AVAILABLE and isinstance(x, np.ndarray) and hasattr(x, 'get'):
        return x.get()
    return x