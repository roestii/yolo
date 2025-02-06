#ifndef LAYERS_H
#define LAYERS_H

#include "types.h"
#include "f2x2_3x3_convolution.h"
#include "f3x3_2x2_convolution.h"

#define BATCH_SIZE 32
#define NEGATIVE_SLOPE -1e-2

#define PADDING F2x2_3x3FILTER_SIZE - 1

/* #define LEAKY_RELU(x) x >= 0 ? x : x * 1e-2
#define N_PATCHES_DIM(inputSize, kernelSize, padding, kernelStride) \
    ((inputSize - kernelSize + 2 * padding) / kernelStride + 1)

#define N_PATCHES(...) N_PATCHES_DIM(__VA_ARGS__) * N_PATCHES_DIM(__VA_ARGS__)
#define N_PER_PATCH(kernelSize, inputChannels) kernelSize * kernelSize * inputChannels

#define CONV_LAYER(name, inputChannels, inputSize, outputChannels,	\
		   kernelSize, kernelStride, padding)			\
    const int name ## nPatches = N_PATCHES(inputSize, kernelSize, padding, kernelStride); \
    const int name ## nPerPatch = N_PER_PATCH(kernelSize, inputChannels); \
    static float name ## output[outputChannels * name ## nPatches];	\
    static float name ## input[inputChannels * inputSize * inputSize];	\
    static float name ## kernel[outputChannels * inputChannels * kernelSize * kernelSize]; \
    static float name ## patches[name ## nPatches * name ## nPerPatch]; \
    static float name ## dloutput[sizeof(name ## output) / sizeof(float)]; \
    static float name ## dlinput[sizeof(name ## input) / sizeof(float)]; \
    static float name ## dlkernel[sizeof(name ## kernel) / sizeof(float)]; \
    void name ## Forward()						\
    {									\
	imageToColumns(name ## patches, name ## input, inputSize,	\
		       inputChannels, kernelSize, kernelStride);	\
	matmulSlow(name ## kernel, name ## patches, name ## output,	\
		   outputChannels, name ## nPatches, name ## nPerPatch);
    }									\
    void name ## Backward(float* dloutput)				\
    {									\
									\
    }									\
*/


// void convolutionForward(float*, float*, float*, float*, int, int, int, int, int, int);
void convolutionBackward(float*, float*, float*, float*,
			 float*, int, int, int, int);
#endif
