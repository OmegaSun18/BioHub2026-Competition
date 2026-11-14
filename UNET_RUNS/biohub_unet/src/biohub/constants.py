"""Dataset constants that never change between runs."""
import numpy as np

SCALE = np.array([1.625, 0.40625, 0.40625], dtype=np.float32)   # um per voxel (z, y, x)
