#include "layers.h"

#define BATCH_SIZE 32
#define NEGATIVE_SLOPE -1e-2

void convBnActForward3x3s1(float* activation, float* output, float* input, float* patches,
			   float* kernel, int inputChannels, int inputSize,
			   int outputChannels, int kernelSize, int kernelStride,
			   int nPatches, int nPerPatch)
{
    // TODO(louis): Fuse imageToColumns and matmul according to the paper
    // Since we only use 3 x 3 kernels with a stride of 1 the winograd algorithm (lavin & gray) can be nice here.
    imageToColumns(patches, input, inputSize,
		   inputChannels, kernelSize,
		   kernelStride);

    matmulSlow(kernel, patches, output,
	       outputChannels, nPatches,
	       nPerPatch);

    for (int i = 0;
	 i < outputChannels * nPatches;
	 ++i, ++output, ++activation)
    {
	if (*output >= 0)
	{
	    *activation = *output;
	}
	else
	{
	    *activation = *output * NEGATIVE_SLOPE;
	}
    }
}

// TODO(louis): introduce a bias?
void convBnActBackward3x3s1(float* dlkernel, float* dlinput, float* dlpatches
			    float* dlactivation, float* dloutput, float* output, float* kernel, float* patches,
			    int inputChannels, int inputSize, int outputChannels,
			    int kernelSize, int kernelStride, int nPatches, int nPerPatch)
{
    // NOTE(louis):
    // * dloutput: outputChannels, nPatches
    // * kernel: outputChannels, nPerPatch
    // * patches: nPerPatch, nPatches
    // * input: inputSize, inputSize

    for (int i = 0;
	 i < outputChannels * nPatches;
	 ++i, ++dlactivation, ++output, ++dloutput)
    {
	if (*output >= 0)
	{
	    *dloutput = *dlactivation;
	}
	else
	{
	    *dloutput = *dlactivation * NEGATIVE_SLOPE;
	}
    }

    // TODO(louis): Fuse matmul and col2im, according to paper
    if (dlinput)
    {
	matmulATransposedB(kernel, dloutput, dlpatches,
			   nPerPatch, nPatches, outputChannels);
	columnsToImage(dlinput, dlpatches, inputSize,
		       inputChannels, kernelSize, kernelStride);
    }

    matmulABTransposed(dloutput, patches, dlkernel,
		       outputChannels, nPerPatch, nPatches);
    
}
