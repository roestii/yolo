#include "layers.h"

#define BATCH_SIZE 32
#define NEGATIVE_SLOPE -1e-2

void convolutionForward(float* output, float* input, float* patches,
			float* kernel, int inputChannels, int inputSize,
			int outputChannels, int kernelSize, int kernelStride,
			int nPatches, int nPerPatch)
{
    imageToColumns(patches, input, inputSize,
		   inputChannels, kernelSize,
		   kernelStride);
    
    matmulSlow(kernel, patches, output,
	       outputChannels, nPatches,
	       nPerPatch);

    for (int i = 0;
	 i < outputChannels * nPatches;
	 ++i, ++output)
    {
	if (*output < 0)
	{
	    *output *= NEGATIVE_SLOPE;
	}
    }
}

// TODO(louis): introduce a bias?
void convolutionBackward(float* dlkernel, float* dlinput, float* dlpatches,
			 float* dloutput, float* kernel, float* patches,
			 int inputChannels, int inputSize, int outputChannels,
			 int kernelSize, int kernelStride, int nPatches, int nPerPatch)
{
    // NOTE(louis):
    // * dloutput: outputChannels, nPatches
    // * kernel: outputChannels, nPerPatch
    // * patches: nPerPatch, nPatches
    // * input: inputSize, inputSize

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
