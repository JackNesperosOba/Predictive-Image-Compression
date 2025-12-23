import numpy as np
def calcul_PAE(img, img_desquantitzat):
    PAE = np.max(img - img_desquantitzat)
    return PAE

def calcul_MSE(img, img_desquantitzat):
    img = img.astype(np.float64)
    img_desquantitzat = img_desquantitzat.astype(np.float64)
    
    MSE = np.mean((img - img_desquantitzat) ** 2)
    return MSE

def calcul_PSNR(MSE, bits):
    # Caso MSE = 0 → PSNR infinito sin warnings
    if MSE == 0:
        return float('inf')

    # Desactiva warnings de divide by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        PSNR = 10 * np.log10((((2**bits)-1)**2) / MSE)
    return PSNR