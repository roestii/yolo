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
    /* if (dlinput)
    {
	matmulATransposedB(kernel, dloutput, dlpatches,
			   nPerPatch, nPatches, outputChannels);
	columnsToImage(dlinput, dlpatches, inputSize,
		       inputChannels, kernelSize, kernelStride);
    }

    matmulABTransposed(dloutput, patches, dlkernel,
    outputChannels, nPerPatch, nPatches); */

    // TODO(louis): dL/dk

    for (int kernel = 0;
	 kernel < kernels;
	 ++kernel)
    {
	// TODO(louis): This is for each of the output channels of dL/dOs
	for (int channel = 0;
	     channel < channels;
	     ++channel)
	{
	    // TODO(louis): This is for each input channel, the kernel one contributes to output channel one, it's individual kernels contribute
	    // to the output using the corresponding input channel. Thus, kernel 1 at channel 1 is the convolution I1 * dL/dO1, at channel
	    // 2 I2 * dL/dO1, at channel 3 I3 * dL/dO1 and so on.

	    // Calculate dL/dk[kernel, channel] using I[channel] conv dL/dO[kernel] with the f(3x3, 2x2) convolution
	}
    }

    // TODO(louis): dL/dI
    for (int channel = 0;
	 channel < channels;
	 ++channel)
    {
	for (int kernel = 0;
	     kernel < kernels;
	     ++kernel)
	{
	    // The output/kernel channel
	    // TODO(louis): Calculate dL/dI using the kernel at output channel kernel with the input channel channel
	    // and add them up to dL/dI for each output channel (each input channel contributes to each output channel using the
	    // corresponding part of the filter that is associated with the input channel, thus we have to iterate over all output channels
	    // for each input channel and adding the losses up.
	    // dL/dI[channel] += flipped(k[kernel, channel]) full conv dL/dO[kernel]
	}
    }
    
}
