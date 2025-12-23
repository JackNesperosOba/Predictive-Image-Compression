import numpy as np
from numba import njit

def predictor(img, components, cols):
    img_int = img.astype(np.int32)
    Ri = np.zeros_like(img_int)
    Ri[:, 0, :] = img_int[:, 0, :]
    Ri[:, 1:, :] = img_int[:, 1:, :] - img_int[:, :-1, :]
    
    return Ri.astype(np.int32)

def descompressor(pred, components, cols):
    img_rec = np.cumsum(pred.astype(np.int32), axis=1)
    return img_rec.astype(np.int32)

def predictor_med(img, components, cols):
    img = img.astype(np.int32)
    rows, cols, comp = img.shape
    Ri = np.zeros_like(img)

    # bordes
    Ri[0, :, :] = img[0, :, :]
    Ri[:, 0, :] = img[:, 0, :]

    for i in range(1, rows):
        for j in range(1, cols):
            N  = img[i-1, j]
            W  = img[i, j-1]
            NW = img[i-1, j-1]

            pred = np.where(
                NW >= np.maximum(N, W), np.minimum(N, W),
                np.where(
                    NW <= np.minimum(N, W), np.maximum(N, W),
                    N + W - NW
                )
            )

            Ri[i, j] = img[i, j] - pred

    return Ri.astype(np.int32)


def descompressor_med(residual, components, cols):
    residual = residual.astype(np.int32)
    rows, cols, comp = residual.shape
    img = np.zeros_like(residual)

    img[0, :, :] = residual[0, :, :]
    img[:, 0, :] = residual[:, 0, :]

    for i in range(1, rows):
        for j in range(1, cols):
            N  = img[i-1, j]
            W  = img[i, j-1]
            NW = img[i-1, j-1]

            pred = np.where(
                NW >= np.maximum(N, W), np.minimum(N, W),
                np.where(
                    NW <= np.minimum(N, W), np.maximum(N, W),
                    N + W - NW
                )
            )

            img[i, j] = residual[i, j] + pred

    return img.astype(np.int32)

@njit(cache=True)
def _med_predictor_core(img):
    rows, cols, comp = img.shape
    Ri = np.zeros_like(img)

    # bordes
    Ri[0, :, :] = img[0, :, :]
    Ri[:, 0, :] = img[:, 0, :]

    for i in range(1, rows):
        for j in range(1, cols):
            for c in range(comp):
                N  = img[i-1, j, c]
                W  = img[i, j-1, c]
                NW = img[i-1, j-1, c]

                if NW >= max(N, W):
                    pred = min(N, W)
                elif NW <= min(N, W):
                    pred = max(N, W)
                else:
                    pred = N + W - NW

                Ri[i, j, c] = img[i, j, c] - pred

    return Ri


@njit(cache=True)
def _med_inverse_core(res):
    rows, cols, comp = res.shape
    img = np.zeros_like(res)

    img[0, :, :] = res[0, :, :]
    img[:, 0, :] = res[:, 0, :]

    for i in range(1, rows):
        for j in range(1, cols):
            for c in range(comp):
                N  = img[i-1, j, c]
                W  = img[i, j-1, c]
                NW = img[i-1, j-1, c]

                if NW >= max(N, W):
                    pred = min(N, W)
                elif NW <= min(N, W):
                    pred = max(N, W)
                else:
                    pred = N + W - NW

                img[i, j, c] = res[i, j, c] + pred

    return img


def predictor_med2(img, components, cols):
    return _med_predictor_core(img.astype(np.int32))


def descompressor_med2(residual, components, cols):
    return _med_inverse_core(residual.astype(np.int32))

