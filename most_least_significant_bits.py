import numpy as np

def split_lsb(residual, k=1):
    r = residual.astype(np.int32)
    msb = r >> k
    lsb = r & ((1 << k) - 1)
    return msb, lsb

def merge_lsb(msb, lsb, k=1):
    return (msb << k) | lsb