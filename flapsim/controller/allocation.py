"""Replace this local linear allocation with a calibrated nonlinear mixer as needed."""
import numpy as np


class LinearAllocator:
    def __init__(self, matrix, trim, velocity_limit):
        self.matrix = np.asarray(matrix, dtype=float)
        self.trim = np.asarray(trim, dtype=float)
        self.velocity_limit = velocity_limit
        if self.matrix.shape != (len(self.trim), 6) or not np.isfinite(self.matrix).all():
            raise ValueError('allocation must be a finite (driven joints, 6) matrix')
        if not np.any(self.matrix):
            raise ValueError('allocation cannot be all zero')
        self.rank = int(np.linalg.matrix_rank(self.matrix))

    def allocate(self, wrench_body, xp=np):
        return xp.clip(xp.asarray(self.trim) + xp.asarray(self.matrix) @ wrench_body,
                       -self.velocity_limit, self.velocity_limit)
