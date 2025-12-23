import argparse
import load_save_file as ld
import entropy as h
import Coder as Cod
import Decoder as Dec
import metrics as mt


# !python cli.py comprimir 03508649.ube16_1_512_512.raw 50 # 50 es el qstep como parametro
# !python cli.py entropia -m espacial 03508649.ube16_1_512_512.raw
# !python cli.py descomprimir compressed.bin
# !python cli.py metricas 03508649.ube16_1_512_512.raw compressed.bin

# cmd entropia
def cmd_entropia(filename, mod):
    rows, cols, components, bits, endian, sign = ld.load_parameters(filename)
    img, dtype = ld.load_file(filename, rows, cols, components, bits, endian, sign)
    
    if mod == "espacial":
        ent = h.entropia_1_espacial(img, components)
    else:
        ent = h.entropia_1_intercanal(img, components)
    
    print(f"Entropía de {filename}: {ent}")

# cmd comprimir
def cmd_comprimir(filename, outfile, qstep):
    print("Ejecutando Codificador()...")
    qstep = int(qstep)
    header, buffer, bps, img_original, bits = Cod.Coder(filename, qstep)   

    ld.save_file_pickle(header, buffer, filename=outfile)
    print(f"Archivo comprimido guardado en: {outfile}")


# cmd descomprimir
def cmd_descomprimir(file_compressed, outfile):
    img = Dec.Decoder(file_compressed)      

    ld.save_file(outfile, img)          
    print(f"Imagen descomprimida guardada en: {outfile}")

# cmd metricas
def cmd_metricas(file_original, file_compressed):

    # Cargar imagen original
    rows, cols, components, bits, endian, sign = ld.load_parameters(file_original)
    img_original, dtype = ld.load_file(file_original, rows, cols, components, bits, endian, sign)

    # Decodificar imagen comprimida
    img_descomprimida = Dec.Decoder(file_compressed)
    
    # Calcular métricas
    PAE = mt.calcul_PAE(img_original, img_descomprimida)
    MSE = mt.calcul_MSE(img_original, img_descomprimida)
    PSNR = mt.calcul_PSNR(MSE, bits)    
    
    # PAE = 0 --> descodificación exacta (sin pérdida)
    # MSE Cuanto mas bajo mejor 
    # PSNR > 50 dB calidad casi perfecta, < 30 dB mala
    
    # MSE bajo + PAE alto --> hubo pocos errores pero graves
    # MSE bajo + PAE bajo --> todo perfecto
    # MSE = 0 --> reconstrucción perfecta (sin pérdidas)
    
    print(f"PAE: {PAE}")
    print(f"MSE: {MSE}") 
    print(f"PSNR: {PSNR}")


def main():
    parser = argparse.ArgumentParser(description="CLI para el compresor")

    sub = parser.add_subparsers(dest="cmd", required=True)

    # Entropía
    p1 = sub.add_parser("entropia")
    p1.add_argument("-m", "--mod_H", default="espacial")
    p1.add_argument("file")

    # Comprimir
    p2 = sub.add_parser("comprimir")
    p2.add_argument("file")
    p2.add_argument("-o", "--outfile", default="compressed.bin")
    p2.add_argument("qstep")

    # Descomprimir
    p3 = sub.add_parser("descomprimir")
    p3.add_argument("file")
    p3.add_argument("-o", "--outfile", default="output.raw")
    
    # Metricas
    p4 = sub.add_parser("metricas")
    p4.add_argument("file_original")
    p4.add_argument("file_compressed")

    args = parser.parse_args()

    if args.cmd == "entropia":
        cmd_entropia(args.file, args.mod_H)

    elif args.cmd == "comprimir":
        cmd_comprimir(args.file, args.outfile, args.qstep)

    elif args.cmd == "descomprimir":
        cmd_descomprimir(args.file, args.outfile)
    
    elif args.cmd == "metricas":
        cmd_metricas(args.file_original, args.file_compressed)


if __name__ == "__main__":
    main()