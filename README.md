# Predictive-Image-Compression
## Predictive Image Compression Project Overview
This project done in the Teoria de Compressio de la Informació(TCI) subject implements a predictive image codec based on spatial prediction and entropy coding, inspired by JPEG-LS principles. The system combines  quantization, MED prediction, residual processing, and arithmetic coding to reduce the bit rate while maintaining high reconstruction quality.
The main objective is to analyze the rate–distortion trade-off, minimizing bits per pixel (bps) while preserving a high PSNR for different quantization steps (qstep).

## How to Use
1. Upload your imagen with raw format in directory Images with name format as Name.Properties.raw, where properties are Unsigned/Signed - Bigendian/Littlendian - e - bits - components - rows - cols, like the images saved in the directory.
2. Execute the graphs.py file and enter the full name of the raw image saved
3. See results of execution

You can do a test with the images already uploaded

## Encoding Pipeline
1. Image loading and parameter extraction
2. Quantization (controlled by qstep)
3. MED spatial prediction
4. Residual processing with MSB/LSB separation
5. Signed-to-unsigned mapping
6. Arithmetic coding
7. Generation of the compressed bitstream with header as compressed.bin

For qstep = 1, the system is fully reversible.

## Decoding Pipeline
The decoder performs the exact inverse operations:
1. Compressed image loading and parameter extraction from header
2. Arithmetic decoding
3. Reverse mapping
4. MSB/LSB merging
5. Inverse MED prediction
6. Dequantization
7. Image reconstruction

## MED Predictor
The MED (Median Edge Detector) predictor uses three neighboring pixels (N, W, NW) to estimate the current pixel. It adapts well to edges and significantly reduces residual variance. The predictor and its inverse are implemented with Numba for efficiency.
Quantization and MSB/LSB Separation
qstep = 1: lossless reconstruction (PSNR = ∞)
qstep > 1: controlled lossy compression
Residuals can be split into most significant bits (MSB) and least significant bits (LSB). This reversible separation concentrates most of the signal energy in the MSB, enabling a significant reduction in bps while preserving image quality.

## Evaluation Metrics
Bits per pixel (bps),
MSE (Mean Square Erro),
PSNR (Peak Signal-to-Noise Ratio).
These metrics are evaluated as a function of qstep.

## Requirements
Python 3.x,
NumPy,
Numba,
Matplotlib

## Conclusion
This project shows how predictive coding, combined with entropy coding and bit-level residual processing, can achieve efficient image compression. The MSB/LSB separation is particularly effective in reducing bps while maintaining PSNR, providing a solid basis for further rate–distortion optimization
