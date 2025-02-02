#include "f3x3_2x2_convolution.h"

// According to Lavin and Gray, the matrics B^T, A^T, and G
static float B[] = {
     1, 0,  0,  0,
     0, 1, -1,  1,
    -1, 1,  1,  0,
     0, 0,  0, -1
};

static float B_T[] = {
    1,  0, -1,  0,
    0,  1,  1,  0,
    0, -1,  1,  0,
    0,  1,  0, -1
};

static float G[] = {
    1.0,  0.0, 0.0,
    0.5,  0.5, 0.5,
    0.5, -0.5, 0.5,
    0.0,  0.0, 1.0
};

static float G_T[] = {
    1.0, 0.5,  0.5, 0.0,
    0.0, 0.5, -0.5, 0.0,
    0.0, 0.5,  0.5, 1.0
};

static float A[] = {
    1,  0,
    1,  1,
    1, -1,
    0, -1
};

static float A_T[] = {
    1, 1,  1,  0,
    0, 1, -1, -1
};

static float uTmp[F3x3_2x2INPUT_TILE_SIZE * F3x3_2x2INPUT_TILE_SIZE];
static float vTmp[F3x3_2x2INPUT_TILE_SIZE * F3x3_2x2INPUT_TILE_SIZE];
static float outputTmp[F3x3_2x2OUTPUT_TILE_SIZE * F3x3_2x2INPUT_TILE_SIZE];
static float d[F3x3_2x2INPUT_TILE_SIZE * F3x3_2x2INPUT_TILE_SIZE];
static float yTmp[F3x3_2x2OUTPUT_TILE_SIZE * F3x3_2x2OUTPUT_TILE_SIZE];
static float AmTmp[F3x3_2x2OUTPUT_TILE_SIZE * F3x3_2x2INPUT_TILE_SIZE];

// TODO(louis): We need matrices U, V, and M that fit all of the us, vs, and ms.
// Should we determine the maximum size needed for U and V for every convolution,
// and just hand them to the function? Probably yes

static void tiledCopy(float* tile, float* input, int inputSize)
{
    // TODO(louis): We could hand roll this completely and use simd registers
    for (int row = 0;
	 row < F3x3_2x2INPUT_TILE_SIZE;
	 ++row, input += inputSize)
    {
	for (int col = 0;
	     col < F3x3_2x2INPUT_TILE_SIZE;
	     ++col, ++tile)
	{
	    *tile = *(input + col);
	}
    }
}

static void untile(float* output, float* input, int outputSize)
{
    float* colStart = output;
    for (int i = 0;
	 i < F3x3_2x2OUTPUT_TILE_SIZE;
	 ++i, colStart += outputSize)
    {
	for (int j = 0;
	     j < F3x3_2x2OUTPUT_TILE_SIZE;
	     ++j, ++input)
	{
	    *(colStart + j) = *input;
	}
    }
}

static void tileElementReduce2d(float* output, float* input, int nDim1,
				 int nDim2, int tileElement)
{
    input += tileElement;
    for (int row = 0;
	 row < nDim1;
	 ++row)
    {
	for (int col = 0;
	     col < nDim2;
	     ++col, ++output,
		 input += F3x3_2x2INPUT_TILE_SIZE * F3x3_2x2INPUT_TILE_SIZE)
	{
	    *output = *input;
	}
    }
}

void f3x3_2x2Convolution(float* Y, float* U, float* V, float* M,
			 float* Utmp, float* Vtmp, float* Mtmp,
			 float* input, float* filter, int inputSize,
			 int channels, int kernels, int tiles)
{
    // TODO(louis): write a reference implementation that works in python

    
    // TODO(louis): How do we have to lay out the memory for the filters, and the input?
    // What if we recompose the input channels x tiles x 4 x 4, we should in fact do that, as the output is of that
    // same shape as well

        
    // NOTE(louis): the input is of size channels * inputSize * inputSize
    // we use output tiles of size 2x2 and a filter of size 3x3 to compute
    // the convolution F(2x2, 3x3) according to the algorithm proposed
    // by Lavin and Gray in https://arxiv.org/pdf/1509.09308.
    // The number of output tiles is given by (inputSize**2 / 4) tiles
    // for a 2x2 tile. The input tile size has to be m + r - 1 for F(mxm, rxr)
    // Thus it is 4 in this case.

    // M: k * t * 4 * 4
    // U: k * c * 4 * 4
    // V: c * t * 4 * 4
    // Utmp: k * c
    // Vtmp: c * t

    // For this to be valid the inputSize has to be divisible without remainder.
    int pxlsPerChannel = inputSize * inputSize;
    // int tiles = pxlsPerChannel >> 2;
    float* g = filter;
    float* tilePtr = input;
    float* columnTilePtr;
    float* u = U;
    float* v = V;
    float* m = M;
    
    for (int kernel = 0;
	 kernel < kernels;
	 ++kernel)
    {
	for (int channel = 0;
	     channel < channels;
	     ++channel,
		 g += F3x3_2x2FILTER_SIZE * F3x3_2x2FILTER_SIZE,
		 u += F3x3_2x2INPUT_TILE_SIZE * F3x3_2x2INPUT_TILE_SIZE)
	{
	    // u = G * g[k,c] * G^T
	    // G: 4x3, g[k,c]: 3x3 -> u: 4x4
	    // U: k * c * 4 * 4
	    matmulSlow(G, g, uTmp,
		       F3x3_2x2INPUT_TILE_SIZE, F3x3_2x2FILTER_SIZE,
		       F3x3_2x2FILTER_SIZE);

	    matmulSlow(uTmp, G_T, u,
		       F3x3_2x2INPUT_TILE_SIZE, F3x3_2x2INPUT_TILE_SIZE,
		       F3x3_2x2FILTER_SIZE);
	}
    }

    int row, col;
    for (int channel = 0;
	 channel < channels;
	 ++channel,
	     tilePtr += F3x3_2x2OUTPUT_TILE_SIZE * inputSize)
    {
	for (row = 0;
	     row < inputSize - F3x3_2x2INPUT_TILE_SIZE + 1;
	     row += F3x3_2x2OUTPUT_TILE_SIZE,
		 tilePtr += F3x3_2x2TILE_OVERLAP + (F3x3_2x2TILE_OVERLAP - 1) * inputSize)
	{
	    for (col = 0;
		 col < inputSize - F3x3_2x2INPUT_TILE_SIZE + 1;
		 col += F3x3_2x2OUTPUT_TILE_SIZE,
		     v += F3x3_2x2INPUT_TILE_SIZE * F3x3_2x2INPUT_TILE_SIZE,
		     tilePtr += F3x3_2x2TILE_OVERLAP
		)
	    {
		// TODO(louis): Make these matrix multiplications fast
		// as they are of known size, with something like schwartz vaknin
		// (recursive 2x2 matmuls)
	    
		// v = B^T * d[c,t] * B
		// B^T: 4x4, d[c,t]: 4x4 -> v: 4x4
		// V: c * t * 4 * 4

		tiledCopy(d, tilePtr, inputSize);
		matmulSlow(B_T, d, vTmp, F3x3_2x2INPUT_TILE_SIZE,
			   F3x3_2x2INPUT_TILE_SIZE, F3x3_2x2INPUT_TILE_SIZE);
		matmulSlow(vTmp, B, v, F3x3_2x2INPUT_TILE_SIZE,
			   F3x3_2x2INPUT_TILE_SIZE, F3x3_2x2INPUT_TILE_SIZE);
	    }
	}
    }

    // TODO(louis): Somehow the ordering of M is wrong, I'm worried this is gonna be a hard one
    // we have to figure out the correct matrix multiply here
    for (int tileElement = 0;
	 tileElement < F3x3_2x2INPUT_TILE_SIZE * F3x3_2x2INPUT_TILE_SIZE;
	 ++tileElement)
    {
	m = M + tileElement;
	// M[:,:,i,j] = U[:,:,i,j] * V[:,:,i,j]
	// M[eps, ups] = U[eps, ups] * V[eps, ups]
	// U[eps, ups]: k * c, V[eps, ups]: c * t
	    
	// M: k * t * 4 * 4

	tileElementReduce2d(Utmp, U, kernels, channels, tileElement);
	tileElementReduce2d(Vtmp, V, channels, tiles, tileElement);
	matmulSlow(Utmp, Vtmp, Mtmp, kernels, tiles, channels);

	float* mTmp = Mtmp;
	for (int k = 0;
	     k < kernels;
	     ++k)
	{
	    for (int t = 0;
		 t < tiles;
		 ++t,
		     m += F3x3_2x2INPUT_TILE_SIZE * F3x3_2x2INPUT_TILE_SIZE,
		     ++mTmp)
	    {
		*m = *mTmp;
	    }
	}
    }

    m = M;
    float* yColStart = Y;
    for (int kernel = 0;
	 kernel < kernels;
	 ++kernel)
    {
	for (int r = 0;
	     r < row;
	     r += F3x3_2x2OUTPUT_TILE_SIZE,
		 yColStart += F3x3_2x2OUTPUT_TILE_SIZE * col)
	{
	    for (int c = 0;
		 c < col;
		 c += F3x3_2x2OUTPUT_TILE_SIZE,
		     m += F3x3_2x2INPUT_TILE_SIZE * F3x3_2x2INPUT_TILE_SIZE)
	    {
		// Y[k,t] = A_T * m * A
		// A: 4x2, m: 4x4: Y[k,t]: 2x2 (F(2x2, 3x3))
		matmulSlow(A_T, m, AmTmp, F3x3_2x2OUTPUT_TILE_SIZE,
			   F3x3_2x2INPUT_TILE_SIZE, F3x3_2x2INPUT_TILE_SIZE);
		matmulSlow(AmTmp, A, yTmp, F3x3_2x2OUTPUT_TILE_SIZE,
			   F3x3_2x2OUTPUT_TILE_SIZE, F3x3_2x2INPUT_TILE_SIZE);
		untile(yColStart + c, yTmp, col);
	    }
	}
    }
}
