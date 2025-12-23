import numpy as np
# mapping.py - VERSIÓN CORREGIDA

def mapping_unsigned(pred):
    """
    Mapping: signed -> unsigned
    Positivos: 1,2,3,4,... -> 1,3,5,7,...
    Negativos: -1,-2,-3,... -> 2,4,6,...
    Cero: 0 -> 0
    """
    pred_int = pred.astype(np.int32)
    mapped = np.where(pred_int > 0, 
                      2 * pred_int - 1,      # Positivos
                      -2 * pred_int)          # Negativos y cero
    return mapped.astype(np.int32)

def rmapping_unsigned(mapped):
    """
    Inverse mapping: unsigned -> signed
    Impares: 1,3,5,7,... -> 1,2,3,4,...
    Pares: 0,2,4,6,... -> 0,-1,-2,-3,...
    """
    mapped_int = mapped.astype(np.int32)
    pred = np.where(mapped_int % 2 == 1,
                    (mapped_int + 1) // 2,   # Impares -> positivos
                    -(mapped_int // 2))       # Pares -> negativos/cero
    return pred.astype(np.int32)

def sub_min(pred):
    min_val = np.min(pred)
    mapped = pred.astype(np.int32) - min_val
    
    return mapped, min_val
        
def rsub_min(pred, min_val):
    rmapped = pred.astype(np.int32) + min_val
    
    return rmapped

def signe_magnitud(pred):
    pred_int = pred.astype(np.int32)
    signe = (pred_int < 0).astype(np.int16)
    mapped = np.where(pred_int > 0, pred_int, -pred_int).astype(np.int16)
    
    return mapped, signe

def rsigne_magnitud(magnitud, signe):
    pred_reconstructed = np.where(signe == 0, magnitud, -magnitud)
    
    return pred_reconstructed.astype(np.int16)

# mapping.py

def mapping_rice(residual):
    r = residual.astype(np.int32)
    mapped = np.empty_like(r, dtype=np.uint32)

    pos = r >= 0
    mapped[pos] = 2 * r[pos]
    mapped[~pos] = -2 * r[~pos] - 1

    return mapped


def rmapping_rice(mapped):
    m = mapped.astype(np.int32)
    r = np.empty_like(m)

    even = (m % 2) == 0
    r[even] = m[even] // 2
    r[~even] = -(m[~even] + 1) // 2

    return r.astype(np.int32)