#include "layers.h"


/*
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
*/

static void flip(float* dest, float* src)
{
    for (int i = 0;
	 i < F2x2_3x3FILTER_SIZE * F2x2_3x3FILTER_SIZE;
	 ++i, ++src, --dest)
    {
	*dest = *src;
    }
}

static void pad(float* output, float* input, int inputChannels, int inputSize, int outputSize)
{
    // The destination has to be of size (4 * PADDING * sizeof(input) / sizeof(float))
    // input: inputChannels, inputsize, inputSize
    // output: inputChannels, inputSize + 2 * PADDING, inputSize + 2 * PADDING
    // The padding is defined as filterSize - 1
     for (int c = 0;
	 c < inputChannels;
	 ++c, output += PADDING * outputSize)
    {
	for (int i = 0;
	     i < inputSize;
	     ++i, output += outputSize)
	{
	    for (int j = 0;
		 j < inputSize;
		 ++j, ++input)
	    {
		*(output + j) = *input;
	    }
	}
    }
}

// TODO(louis): introduce a bias?
static float dloutputTile[F2x2_3x3OUTPUT_TILE_SIZE * F2x2_3x3OUTPUT_TILE_SIZE];
static float dlinputTile[F2x2_3x3OUTPUT_TILE_SIZE * F2x2_3x3OUTPUT_TILE_SIZE];
// TODO(louis): Maybe get rid of the extra memory for the padding.
static float paddeddloutputTile[F2x2_3x3INPUT_TILE_SIZE * F2x2_3x3INPUT_TILE_SIZE];
static float inputTile[F2x2_3x3INPUT_TILE_SIZE * F2x2_3x3INPUT_TILE_SIZE];
static float dlkernelTmp[F2x2_3x3FILTER_SIZE * F2x2_3x3FILTER_SIZE];
static float flippedKernel[F2x2_3x3FILTER_SIZE * F2x2_3x3FILTER_SIZE];

void convolutionBackward(float* dlkernel, float* dlinput, float* dloutput, float* paddeddloutput, float* kernel,
			 float* input, int inputChannels, int inputSize, int outputChannels, int outputSize)
{
    float* inputTileStartPtr, * dloutputTileStartPtr;
    float* dloutputChannelStartPtr = dloutput;
    float* thing = dlkernel;
    
    // TODO(louis): dL/dk
    for (int outputChannel = 0;
	 outputChannel < outputChannels;
	 ++outputChannel,
	     dloutputChannelStartPtr += outputSize * outputSize)
    {
	inputTileStartPtr = input;
	// TODO(louis): This is for each of the output channels of dL/dOs
	for (int inputChannel = 0;
	     inputChannel < inputChannels;
	     ++inputChannel,
		 inputTileStartPtr += F2x2_3x3TILE_OVERLAP * inputSize,
		 dlkernel += F2x2_3x3FILTER_SIZE * F2x2_3x3FILTER_SIZE)
	{
	    dloutputTileStartPtr = dloutputChannelStartPtr;
	    // TODO(louis): This is for each input channel, the kernel one contributes to output channel one, it's individual kernels contribute
	    // to the output using the corresponding input channel. Thus, kernel 1 at channel 1 is the convolution I1 * dL/dO1, at channel
	    // 2 I2 * dL/dO1, at channel 3 I3 * dL/dO1 and so on.

	    // Calculate dL/dk[kernel, channel] using I[channel] conv dL/dO[kernel] with the f(3x3, 2x2) convolution (we also have to split up the
	    // dL/dO into 2x2 tiles

	    for (int row = 0;
		 row < outputSize;
		 row += F2x2_3x3OUTPUT_TILE_SIZE,
		     inputTileStartPtr += F2x2_3x3TILE_OVERLAP + (F2x2_3x3TILE_OVERLAP - 1) * inputSize,
		     dloutputTileStartPtr += (F2x2_3x3OUTPUT_TILE_SIZE - 1) * outputSize)
	    {
		for (int col = 0;
		     col < outputSize;
		     col += F2x2_3x3OUTPUT_TILE_SIZE,
			 inputTileStartPtr += F2x2_3x3TILE_OVERLAP,
			 dloutputTileStartPtr += F2x2_3x3OUTPUT_TILE_SIZE)
		{
		    // Copy the input and output into the local tile arrays.
		    float* dest = inputTile;
		    float* src = inputTileStartPtr;
		    for (int i = 0;
			 i < F2x2_3x3INPUT_TILE_SIZE;
			 ++i, src += inputSize)
		    {
			for (int j = 0;
			     j < F2x2_3x3INPUT_TILE_SIZE;
			     ++j, ++dest)
			{
			    *dest = *(src + j);
			}
		    }

		    dest = dloutputTile;
		    src = dloutputTileStartPtr;
		    for (int i = 0;
			 i < F2x2_3x3OUTPUT_TILE_SIZE;
			 ++i, src += outputSize)
		    {
			for (int j = 0;
			     j < F2x2_3x3OUTPUT_TILE_SIZE;
			     ++j, ++dest)
			{
			    *dest = *(src + j);
			}
		    }

		    f3x3_2x2SingleTileConvolution(dlkernelTmp, inputTile, dloutputTile);
		    // Add the resulting 3x3 matrix to the existing kernel.
		    for (int i = 0;
			 i < F2x2_3x3FILTER_SIZE * F2x2_3x3FILTER_SIZE;
			 ++i)
		    {
		        dlkernel[i] += dlkernelTmp[i];
		    }
		}
	    }
	}
    }

    int paddeddloutputSize = outputSize + 2 * PADDING;
    pad(paddeddloutput, dloutput, outputChannels, outputSize, paddeddloutputSize);
    // TODO(louis): dL/dI
    float* kernelPtr;
    float* paddeddloutputTileStartPtr;
    float* dlinputTileStartPtr;
    for (int inputChannel = 0;
	 inputChannel < inputChannels;
	 ++inputChannel,
	     kernel += F2x2_3x3FILTER_SIZE * F2x2_3x3FILTER_SIZE,
	     dlinput += inputSize * inputSize)
    {
	kernelPtr = kernel;
	paddeddloutputTileStartPtr = paddeddloutput;
	for (int outputChannel = 0;
	     outputChannel < outputChannels;
	     ++outputChannel,
		 kernelPtr += inputChannels * F2x2_3x3FILTER_SIZE * F2x2_3x3FILTER_SIZE,
		 paddeddloutputTileStartPtr += F2x2_3x3TILE_OVERLAP * paddeddloutputSize)
	{
	    // The output/kernel channel
	    // TODO(louis): Calculate dL/dI using the output at outputChannel and the kernel at outputChannel, inputChannel
	    // and add them up to dL/dI for each output channel (each input channel contributes to each output channel using the
	    // corresponding part of the filter that is associated with the input channel, thus we have to iterate over all output channels
	    // for each input channel and adding the losses up.
	    // dL/dI[channel] += padded(dL/doutput, r - 1) * flipped(k[outputChannel, inputChannel], 180)
	    dlinputTileStartPtr = dlinput;
	    flip(flippedKernel, kernelPtr);
	    for (int row = 0;
		 row < paddeddloutputSize;
		 row += F2x2_3x3TILE_OVERLAP,
		     paddeddloutputTileStartPtr += F2x2_3x3TILE_OVERLAP + (F2x2_3x3TILE_OVERLAP - 1) * paddeddloutputSize,
		     dlinputTileStartPtr += (F2x2_3x3OUTPUT_TILE_SIZE - 1) * inputSize)
	    {
		for (int col = 0;
		     col < paddeddloutputSize;
		     col += F2x2_3x3TILE_OVERLAP,
			 paddeddloutputTileStartPtr += F2x2_3x3TILE_OVERLAP,
			 dlinputTileStartPtr += F2x2_3x3OUTPUT_TILE_SIZE)
		{
		    float* src = paddeddloutputTileStartPtr;
		    // TODO(louis): Maybe try to use the padding here...
		    float* dest = paddeddloutputTile;
		    for (int i = 0;
			 i < F2x2_3x3INPUT_TILE_SIZE;
			 ++i, src += paddeddloutputSize)
		    {
			for (int j = 0;
			     j < F2x2_3x3INPUT_TILE_SIZE;
			     ++j, ++dest)
			{
			    *dest = *(src + j);
			}
		    }

		    f2x2_3x3SingleTileConvolution(dlinputTile, paddeddloutputTile, flippedKernel);
		    // Untile the resulting convolution
		    dest = dlinputTileStartPtr;
		    src = dlinputTile;
		    for (int i = 0;
			 i < F2x2_3x3OUTPUT_TILE_SIZE;
			 ++i, dest += inputSize)
		    {
			for (int j = 0;
			     j < F2x2_3x3OUTPUT_TILE_SIZE;
			     ++j, ++src)
			{
			    *(dest + j) = *src;
			}
		    }
		}
	    }
	}
    }
}
