import os
import matplotlib.pyplot as plt
import load_save_file as ld
import Coder as Cod
import Decoder as Dec
import metrics as mt

def graficas(file_original, qstep):

    # Codificar
    header, buffer, bps, img_original, bits = Cod.Coder(file_original, qstep)

    file_compressed = ld.save_file_pickle(header, buffer, filename="compressed.bin")
    
    # Decodificar imagen comprimida
    img_descomprimida, lsb = Dec.Decoder(file_compressed)
        
    # Calcular métricas
    MSE = mt.calcul_MSE(img_original, img_descomprimida)
    PSNR = mt.calcul_PSNR(MSE, bits)    
    
    return bps, PSNR, MSE

IMAGE_DIR = os.path.join("Images")

nombre_imagen = input("Image name: ")
file_original = os.path.join(IMAGE_DIR, nombre_imagen)

if not os.path.isfile(file_original):
    raise FileNotFoundError(f"File not found: {file_original}")

QSTEPS = []
BPS_list = []
PSNR_list = []

print("\n QSTEP \t BPS \t PSNR \t MSE")
for qstep in range(1, 51):
    bps, PSNR, MSE = graficas(file_original, qstep)

    # Convertir punto a coma
    bps_str = str(bps).replace('.', ',')
    psnr_str = str(PSNR).replace('.', ',')
    mse_str = str(MSE).replace('.', ',')

    print(f"{qstep}\t{bps_str}\t{psnr_str}\t{mse_str}")

    QSTEPS.append(qstep)
    BPS_list.append(bps)
    PSNR_list.append(PSNR)


plt.figure(figsize=(7,5), dpi=150)
plt.plot(BPS_list, PSNR_list, linewidth=1.5)
plt.xlabel("Bits por símbolo (bps)")
plt.ylabel("PSNR (dB)")
plt.title("PSNR vs BPS (alta precisión)")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

plt.figure(figsize=(7,5), dpi=150)
plt.plot(QSTEPS, BPS_list, linewidth=1.5)
plt.xlabel("Qstep")
plt.ylabel("Bits por símbolo (bps)")
plt.title("BPS vs Qstep")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

