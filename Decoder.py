import load_save_file as ld
import quantizer as quant
import predictor as pred
import mapping as mapp
import arithmetic_coder as ca
import most_least_significant_bits as sb

def Decoder(file_compressed):
    # Cargar archivo comprimido
    header, buffer = ld.load_file_pickle(filename=file_compressed)

    rows, cols, components, bits, N, qstep, endian, cum_freq, sign, dtype, lsb = header.values()

    # Decodificación aritmética
    magnitud_decoded = ca.Descodificador(
        N, cum_freq, buffer, dtype, shape=(rows, cols, components)
    )

    # Reverse mapping
    rmapp = mapp.rmapping_unsigned(magnitud_decoded)
    
    #Merge most and least significant bits
    rmapp = sb.merge_lsb(rmapp, lsb, k=1)
    
    # Reconstruir mediante predictor inverso
    residus = pred.descompressor_med2(rmapp, components, cols)
    
    # Descuantización
    img_decoded = quant.desquantitzar_normal(residus, qstep)

    return img_decoded, lsb