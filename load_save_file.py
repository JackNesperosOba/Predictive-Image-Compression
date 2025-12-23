import numpy as np
import re
import pickle


def load_parameters(filename):
    #'Landsat_agriculture.ube16_6_1024_1024.raw'
    # '03508649.ube16_1_512_512.raw'
    pattern = r"\.(\w+?)_(\d+)_(\d+)_(\d+)\.raw$"
    
    match = re.search(pattern, filename)

    dataType_str, components, rows, cols = match.groups()

    if dataType_str.lower().startswith("s"):
        sign = "s"
    else:
        sign = "u"
        
    if "b" in dataType_str.lower():
        endian = "b"
    elif "l" in dataType_str.lower():
        endian = "l"
    else:
        endian = "l"  # por defecto

    bits = re.search(r"(\d+)", dataType_str)
    bits = int(bits.group(1)) if bits else 8

    return int(rows), int(cols), int(components), bits, endian, sign, 
    

def load_file(filename, rows, cols, components, bits, endian, sign):
    bits = str(bits)
    # aplicar si signed o unsigned
    if bits == "8":
        base_dtype = np.uint8 if sign == "u" else np.int8
    elif bits == "16":
        base_dtype = np.uint16 if sign == "u" else np.int16
    
    # aplicar big o little endian si es de 16 bits
    dtype = base_dtype
    if bits == "16":
        dtype = np.dtype(base_dtype).newbyteorder("<" if endian == "l" else ">")
    # guardar valores de la imagen en array x,y
    img = np.fromfile(filename, dtype=dtype)
    img = img.reshape(rows, cols, components)
    
    return img, dtype

def save_file(filename, img):
    # guardar imagen en .raw
    output_file = "copia_" + filename
    img.tofile(output_file)
    return None

def convert_bits(img, endian, sign):
    base_dtype = np.uint16 if sign == "u" else np.int16
    dtype = np.dtype(base_dtype).newbyteorder("<" if endian == "l" else ">")
    img = np.array(img, dtype=dtype)
    return img

def save_file_pickle(header, buffer, filename):
    with open(filename, "wb") as f:
        pickle.dump({"header": header, "buffer": buffer}, f)
    return filename
    
def save_file_np(header, buffer, filename):
    np.savez(filename, header=header, buffer=buffer)
    return None
    
def load_file_pickle(filename):
    with open(filename, "rb") as f:
        data = pickle.load(f)

    header = data["header"]
    buffer = data["buffer"]
    return header, buffer
    
def load_file_np(filename):
    data = np.load(filename, allow_pickle=True)
    header = data["header"].item()   # recuperar diccionario
    buffer = data["buffer"]
    return header, buffer




