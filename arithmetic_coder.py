import numpy as np
from numba import njit, uint32

# ----------------------------------------------------------
# Numba accelerated functions
# ----------------------------------------------------------

@njit(cache=True)
def encode_step(low, high, c_low, c_high, total):
    range_ = (high - low) + 1
    high = low + (range_ * c_high) // total - 1
    low  = low + (range_ * c_low)  // total
    return low, high

@njit(cache=True)
def renormalize_encoder(low, high):
    """Devuelve código: 0 -> emitir bit, 1 -> underflow, 2 -> salir"""
    if (high ^ low) < 0x80000000:
        return 0
    if (low & 0x40000000) and not (high & 0x40000000):
        return 1
    return 2

@njit(cache=True)
def decode_find_symbol(value, cum):
    """Búsqueda binaria"""
    left, right = 0, len(cum) - 1
    while left < right - 1:
        mid = (left + right) // 2
        if cum[mid] <= value:
            left = mid
        else:
            right = mid
    return left

@njit(cache=True)
def compute_cum_freq_numba(flat, symbols):
    # freq[i] = número de veces que aparece el símbolo i
    freq = np.zeros(symbols, dtype=np.uint32)
    
    for x in flat:
        if 0 <= x < symbols:  # seguridad
            freq[x] += 1
        else:
            raise ValueError(f"Valor fuera de rango: {x}")
    
    # calcular frecuencia acumulada
    cum = np.empty(symbols + 1, dtype=np.uint32)
    s = 0
    cum[0] = 0
    for i in range(symbols):
        s += freq[i]
        cum[i + 1] = s
    
    return cum

# ----------------------------------------------------------
# Buffer dinámico
# ----------------------------------------------------------

@njit(cache=True)
def _ensure_capacity(buffer, min_pos):
    if min_pos < buffer.size:
        return buffer
    new_size = max(buffer.size * 2, min_pos + 1)
    new_buf = np.zeros(new_size, dtype=np.uint8)
    for i in range(buffer.size):
        new_buf[i] = buffer[i]
    return new_buf

# ----------------------------------------------------------
# Bit I/O
# ----------------------------------------------------------

@njit(cache=True)
def write_bit(buffer, byte_pos, bit_pos, cur_byte, bit):
    bit = uint32(bit & 1)
    cur_byte = (cur_byte << 1) | bit
    bit_pos += 1
    if bit_pos == 8:
        if byte_pos >= buffer.size:
            buffer = _ensure_capacity(buffer, byte_pos)
        buffer[byte_pos] = cur_byte
        byte_pos += 1
        cur_byte = 0
        bit_pos = 0
    return buffer, byte_pos, bit_pos, cur_byte

@njit(cache=True)
def flush_byte(buffer, byte_pos, bit_pos, cur_byte):
    if bit_pos > 0:
        if byte_pos >= buffer.size:
            buffer = _ensure_capacity(buffer, byte_pos)
        cur_byte <<= (8 - bit_pos)
        buffer[byte_pos] = cur_byte
        byte_pos += 1
    return buffer, byte_pos, 0, 0

@njit(cache=True)
def read_bit(buffer, byte_pos, bit_pos):
    if byte_pos >= buffer.size:
        return 0, byte_pos, bit_pos
    bit = (buffer[byte_pos] >> (7 - bit_pos)) & 1
    bit_pos += 1
    if bit_pos == 8:
        bit_pos = 0
        byte_pos += 1
    return bit, byte_pos, bit_pos

# ----------------------------------------------------------
# Arithmetic Encoder
# ----------------------------------------------------------

@njit(cache=True)
def arithmetic_encode(flat, cum, buffer):
    low = uint32(0)
    high = uint32(0xFFFFFFFF)
    underflow = 0
    byte_pos = 0
    bit_pos = 0
    cur_byte = 0
    total = cum[-1]

    for s in flat:
        c_low  = cum[s]
        c_high = cum[s + 1]
        low, high = encode_step(low, high, c_low, c_high, total)

        while True:
            action = renormalize_encoder(low, high)
            if action == 0:
                bit = high >> 31
                buffer, byte_pos, bit_pos, cur_byte = write_bit(buffer, byte_pos, bit_pos, cur_byte, bit)
                for _ in range(underflow):
                    buffer, byte_pos, bit_pos, cur_byte = write_bit(buffer, byte_pos, bit_pos, cur_byte, bit ^ 1)
                underflow = 0
            elif action == 1:
                underflow += 1
                low  &= 0x3FFFFFFF
                high |= 0x40000000
            else:
                break

            low = (low << 1) & 0xFFFFFFFF
            high = ((high << 1) | 1) & 0xFFFFFFFF

    # Finish encoding
    underflow += 1
    if low < 0x40000000:
        buffer, byte_pos, bit_pos, cur_byte = write_bit(buffer, byte_pos, bit_pos, cur_byte, 0)
        for _ in range(underflow):
            buffer, byte_pos, bit_pos, cur_byte = write_bit(buffer, byte_pos, bit_pos, cur_byte, 1)
    else:
        buffer, byte_pos, bit_pos, cur_byte = write_bit(buffer, byte_pos, bit_pos, cur_byte, 1)
        for _ in range(underflow):
            buffer, byte_pos, bit_pos, cur_byte = write_bit(buffer, byte_pos, bit_pos, cur_byte, 0)

    buffer, byte_pos, bit_pos, cur_byte = flush_byte(buffer, byte_pos, bit_pos, cur_byte)
    return buffer[:byte_pos], byte_pos  # solo bytes usados

# ----------------------------------------------------------
# Arithmetic Decoder (seguro)
# ----------------------------------------------------------

@njit(cache=True)
def arithmetic_decode_safe(N, cum, buffer):
    low = 0
    high = 0xFFFFFFFF
    code = 0
    byte_pos = 0
    bit_pos = 0
    decoded = np.empty(N, dtype=np.uint32)

    # Inicializar código
    for _ in range(32):
        bit, byte_pos, bit_pos = read_bit(buffer, byte_pos, bit_pos)
        code = (code << 1) | bit

    total = int(cum[-1])

    for i in range(N):
        range_ = high - low + 1
        if range_ == 0:
            range_ = 1  # seguridad extra
        value = ((code - low + 1) * total - 1) // range_

        # Búsqueda binaria
        left, right = 0, len(cum) - 1
        while left < right - 1:
            mid = (left + right) // 2
            if cum[mid] <= value:
                left = mid
            else:
                right = mid
        symbol = left
        decoded[i] = symbol

        c_low  = cum[symbol]
        c_high = cum[symbol + 1]
        high = low + (range_ * c_high) // total - 1
        low  = low + (range_ * c_low)  // total

        # Renormalización
        while True:
            if (high ^ low) < 0x80000000:
                bit, byte_pos, bit_pos = read_bit(buffer, byte_pos, bit_pos)
                code = ((code << 1) & 0xFFFFFFFF) | bit
                low = (low << 1) & 0xFFFFFFFF
                high = ((high << 1) | 1) & 0xFFFFFFFF
            elif (low & 0x40000000) != 0 and (high & 0x40000000) == 0:
                code ^= 0x40000000
                low &= 0x3FFFFFFF
                high |= 0x40000000
                bit, byte_pos, bit_pos = read_bit(buffer, byte_pos, bit_pos)
                code = ((code << 1) & 0xFFFFFFFF) | bit
                low = (low << 1) & 0xFFFFFFFF
                high = ((high << 1) | 1) & 0xFFFFFFFF
            else:
                break

    return decoded

# ----------------------------------------------------------
# Funciones principales
# ----------------------------------------------------------

def Codificador(bits, mapp):
    # Aplanar la imagen a 1D
    flat = np.ascontiguousarray(mapp.reshape(-1).astype(np.uint32))
    N = flat.size

    # Definir SYMBOLS de forma robusta:
    # Al menos 2^bits, pero si hay valores mayores, se ajusta automáticamente
    SYMBOLS = flat.max() + 1

    # Calcular frecuencia acumulada
    cum_freq = compute_cum_freq_numba(flat, SYMBOLS)

    # Crear buffer inicial seguro
    buffer = np.zeros(N * 2, dtype=np.uint8)

    # Codificación aritmética
    buffer, byte_len = arithmetic_encode(flat, cum_freq, buffer)

    # Bits por símbolo aproximado
    bps = (byte_len * 8) / N

    # Debug opcional
    # print(f"Tamaño codificado: {byte_len} bytes") 
    # print(f"Bits por símbolo (aprox): {bps}")

    return N, cum_freq, buffer, bps

def Descodificador(N, cum_freq, buffer, dtype, shape):
    decoded = arithmetic_decode_safe(N, cum_freq, buffer)
    decoded_img = np.array(decoded, dtype=dtype).reshape(shape)
    return decoded_img