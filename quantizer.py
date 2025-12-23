import numpy as np
def quantitzar(img, qstep):
    if qstep == 1:
        return img.astype(np.int32), 1     # identidad exacta

    energia = np.abs(img)
    escala = 1 + energia / (np.max(energia) + 1e-6)
    q_local = qstep * escala

    img_quant = np.sign(img) * np.round(np.abs(img) / q_local)
    return img_quant.astype(np.int32), q_local


def desquantitzar(img_quant, q_local):
    if isinstance(q_local, int) and q_local == 1:
        return img_quant                  # reversibilidad perfecta

    return img_quant * q_local

def quantitzar_biased(residual, qstep):
    """
    Cuantización mid-tread con corrección de sesgo global.
    Minimiza MSE acumulado en códec predictivo.
    """
    r = residual.astype(np.float32)

    if qstep == 1:
        return residual.astype(np.int32), (1, 0.0)

    # cuantización centrada
    q = np.floor(r / qstep + 0.5)

    # sesgo medio de reconstrucción
    bias = np.mean(r - q * qstep)

    return q.astype(np.int32), (qstep, bias)


def desquantitzar_biased(q_residual, params):
    qstep, bias = params

    if qstep == 1:
        return q_residual.astype(np.int32)

    return (q_residual * qstep + bias).astype(np.int32)

def quantitzar_normal(img, qstep):
    imatge_quant = np.sign(img) * (np.round(np.abs(img) / qstep)) #Round, redondear
    # imatge_quant = np.sign(img) * (np.abs(img) // qstep) #Floor, quitar decimales
    # imatge_quant = np.sign(img) * (np.ceil(np.abs(img) / qstep)) #Ceil, si hay deciaml redondear hacia arriba
    
    return imatge_quant

def desquantitzar_normal(img, qstep):
    imatge_desquant = img * qstep
    
    return imatge_desquant






