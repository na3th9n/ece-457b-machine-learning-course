#!/usr/bin/env python3

import numpy as np

NUM_SAMPLES = 200

def getsamples():
    X_xor = np.random.randn(NUM_SAMPLES, 2)
    y_xor = np.logical_xor(X_xor[:, 0] > 0,
                           X_xor[:, 1] > 0)
    y_xor = np.where(y_xor, 1, 0)
    return (X_xor, y_xor)
