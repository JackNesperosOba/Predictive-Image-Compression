import load_save_file as ld
import quantizer as quant
import predictor as pred
import mapping as mapp
import arithmetic_coder as ca
import most_least_significant_bits as sb

def Coder(filename, qstep):
    # Cargar parámetros e imagen
    rows, cols, components, bits, endian, sign = ld.load_parameters(filename)
    img_original, dtype = ld.load_file(filename, rows, cols, components, bits, endian, sign)

    # Cuantización
    img_quant = quant.quantitzar_normal(img_original, qstep)    

    # Predicción + residuos
    img_pred = pred.predictor_med2(img_quant, components, cols)

    
    # Split significant bits
    msb, lsb = sb.split_lsb(img_pred, k=1)
    
    # Mapping unsigned
    img_mapp = mapp.mapping_unsigned(msb)

    # Codificación aritmética
    N, cum_freq, buffer, bps = ca.Codificador(bits, img_mapp)

    # Cabecera
    header = {
        'rows': rows,
        'cols': cols,
        'components': components,
        'bits': bits,
        'N': N,
        'qstep': qstep,
        'endian': endian,
        'cum_freq': cum_freq,
        'sign': sign,
        'dtype': img_mapp.dtype,
        #'qlocal': q_local, #if you want to use another quantizer
        'lsb': lsb
    }

    return header, buffer, bps, img_original, bits
