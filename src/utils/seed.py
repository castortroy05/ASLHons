"""
Utility functions for setting random seeds for reproducibility.
"""

import random
import numpy as np
import tensorflow as tf
import os


def set_seeds(seed: int = 42):
    """
    Set random seeds for reproducibility across all libraries.

    Args:
        seed: Random seed value
    """
    # Python random module
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # TensorFlow
    tf.random.set_seed(seed)

    # Python hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)

    # TensorFlow deterministic operations (may reduce performance)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'

    print(f"✓ Random seeds set to {seed} for reproducibility")


def get_device(device_preference: str = "auto"):
    """
    Get the appropriate device for computation.

    Args:
        device_preference: 'auto', 'cpu', or 'gpu'

    Returns:
        Device name string
    """
    gpus = tf.config.list_physical_devices('GPU')

    if device_preference == "cpu":
        # Force CPU usage
        tf.config.set_visible_devices([], 'GPU')
        print("✓ Using CPU")
        return "/CPU:0"

    elif device_preference == "gpu":
        if gpus:
            try:
                # Enable memory growth to avoid OOM errors
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print(f"✓ Using GPU: {len(gpus)} device(s) available")
                return "/GPU:0"
            except RuntimeError as e:
                print(f"✗ GPU configuration error: {e}")
                print("  Falling back to CPU")
                return "/CPU:0"
        else:
            print("✗ No GPU available, using CPU")
            return "/CPU:0"

    else:  # auto
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print(f"✓ Auto-detected GPU: {len(gpus)} device(s) available")
                return "/GPU:0"
            except RuntimeError as e:
                print(f"✗ GPU configuration error: {e}")
                print("  Falling back to CPU")
                return "/CPU:0"
        else:
            print("✓ No GPU detected, using CPU")
            return "/CPU:0"


def enable_mixed_precision(enabled: bool = True):
    """
    Enable mixed precision training for faster training on modern GPUs.

    Args:
        enabled: Whether to enable mixed precision
    """
    if enabled:
        from tensorflow.keras import mixed_precision
        policy = mixed_precision.Policy('mixed_float16')
        mixed_precision.set_global_policy(policy)
        print("✓ Mixed precision training enabled (float16)")
    else:
        print("✓ Using full precision (float32)")
