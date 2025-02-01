#include "algebra.h"

void matmulSlow(float* a, float* b, float* c, int m, int n, int k)
{
    // NOTE(louis): a is m x k, b is k x n, thus c has to be m x n
    for (int i = 0; i < m; ++i)
    {
	for (int j = 0; j < n; ++j, ++c)
	{
	    *c = 0;
	    for (int l = 0; l < k; ++l)
	    {
		*c += a[i * k + l] * b[l * n + j];
	    }
	}
    }
}

void matmulATransposedB(float* a, float* b, float* c, int m, int n, int k)
{
    // NOTE(louis): a is k x m, b is k x n, thus c is m x n
    for (int i = 0; i < m; ++i)
    {
	for (int j = 0; j < n; ++j, ++c)
	{
	    *c = 0;
	    for (int l = 0; l < k; ++l)
	    {
		*c += a[l * m + i] * b[l * n + j];
	    }
	}
    }
}

void matmulABTransposed(float* a, float* b, float* c, int m, int n, int k)
{
    // NOTE(louis): a is m x k, b is n x k, thus c is m x n
    for (int i = 0; i < m; ++i)
    {
	for (int j = 0; j < n; ++j, ++c)
	{
	    *c = 0;
	    for (int l = 0; l < k; ++l)
	    {
		*c += a[i * k + l] * b[j * k + l];
	    }
	}
    }
}


// According to Lavin and Gray, the matrics B^T, A^T, and G
 
static float F2x2_3x3B[] = {
     1, 0,  0,  0,
     0, 1, -1,  1,
    -1, 1,  1,  0,
     0, 0,  0, -1
};

static float F2x2_3x3B_T[] = {
    1,  0, -1,  0,
    0,  1,  1,  0,
    0, -1,  1,  0,
    0,  1,  0, -1
};


static float F2x2_3x3G[] = {
    1.0,  0.0, 0.0,
    0.5,  0.5, 0.5,
    0.5, -0.5, 0.5,
    0.0,  0.0, 1.0
};

static float F2x2_3x3G_T[] = {
    1.0, 0.5,  0.5, 0.0,
    0.0, 0.5, -0.5, 0.0,
    0.0, 0.5,  0.5, 1.0
};

static float F2x2_3x3A[] = {
    1,  0,
    1,  1,
    1, -1,
    0, -1
};

static float F2x2_3x3A_T[] = {
    1, 1,  1,  0,
    0, 1, -1, -1
};

static float uTmp[F2x2_3x3INPUT_TILE_SIZE * F2x2_3x3INPUT_TILE_SIZE];
static float vTmp[F2x2_3x3INPUT_TILE_SIZE * F2x2_3x3INPUT_TILE_SIZE];
static float outputTmp[F2x2_3x3OUTPUT_TILE_SIZE * F2x2_3x3INPUT_TILE_SIZE];
static float d[F2x2_3x3INPUT_TILE_SIZE * F2x2_3x3INPUT_TILE_SIZE];

// TODO(louis): We need matrices U, V, and M that fit all of the us, vs, and ms.
// Should we determine the maximum size needed for U and V for every convolution,
// and just hand them to the function? Probably yes

void tiledCopy(float* tile, float* input, int inputSize)
{
    // TODO(louis): We could hand roll this completely and use simd registers
    for (int row = 0;
	 row < F2x2_3x3INPUT_TILE_SIZE;
	 ++row, input += inputSize)
    {
	for (int col = 0;
	     col < F2x2_3x3INPUT_TILE_SIZE;
	     ++col, ++tile)
	{
	    *tile = *(input + col);
	}
    }
}

void tileElementReduce2d(float* output, float* input, int nDim1,
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
		 input += F2x2_3x3INPUT_TILE_SIZE * F2x2_3x3INPUT_TILE_SIZE)
	{
	    *output = *input;
	}
    }
}

void f2x2_3x3Convolution(float* output, float* U, float* V, float* M,
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
		 g += F2x2_3x3FILTER_SIZE * F2x2_3x3FILTER_SIZE,
		 u += F2x2_3x3INPUT_TILE_SIZE * F2x2_3x3INPUT_TILE_SIZE)
	{
	    // u = G * g[k,c] * G^T
	    // G: 4x3, g[k,c]: 3x3 -> u: 4x4
	    // U: k * c * 4 * 4
	    matmulSlow(F2x2_3x3G, g, uTmp,
		       F2x2_3x3INPUT_TILE_SIZE, F2x2_3x3FILTER_SIZE,
		       F2x2_3x3FILTER_SIZE);

	    matmulSlow(uTmp, F2x2_3x3G_T, u,
		       F2x2_3x3INPUT_TILE_SIZE, F2x2_3x3INPUT_TILE_SIZE,
		       F2x2_3x3FILTER_SIZE);
	}
    }

    // TODO(louis): collapse these loops
    for (int channel = 0;
	 channel < channels;
	 ++channel,
	     tilePtr += F2x2_3x3OUTPUT_TILE_SIZE * inputSize)
    {
	for (int row = 0;
	     row < inputSize - F2x2_3x3INPUT_TILE_SIZE + 1;
	     row += F2x2_3x3OUTPUT_TILE_SIZE,
		 tilePtr += F2x2_3x3OUTPUT_TILE_SIZE + (F2x2_3x3OUTPUT_TILE_SIZE - 1) * inputSize)
	{
	    for (int col = 0;
		 col < inputSize - F2x2_3x3INPUT_TILE_SIZE + 1;
		 col += F2x2_3x3OUTPUT_TILE_SIZE,
		     v += F2x2_3x3INPUT_TILE_SIZE * F2x2_3x3INPUT_TILE_SIZE,
		     tilePtr += F2x2_3x3OUTPUT_TILE_SIZE
		)
	    {
		// TODO(louis): Make these matrix multiplications fast
		// as they are of known size, with something like schwartz vaknin
		// (recursive 2x2 matmuls)
	    
		// v = B^T * d[c,t] * B
		// B^T: 4x4, d[c,t]: 4x4 -> v: 4x4
		// V: c * t * 4 * 4

		tiledCopy(d, tilePtr, inputSize);
		matmulSlow(F2x2_3x3B_T, d, vTmp, F2x2_3x3INPUT_TILE_SIZE,
			   F2x2_3x3INPUT_TILE_SIZE, F2x2_3x3INPUT_TILE_SIZE);

		matmulSlow(vTmp, F2x2_3x3B, v, F2x2_3x3INPUT_TILE_SIZE,
			   F2x2_3x3INPUT_TILE_SIZE, F2x2_3x3INPUT_TILE_SIZE);
	    }
	}
    }

    // TODO(louis): Somehow the ordering of M is wrong, I'm worried this is gonna be a hard one
    // we have to figure out the correct matrix multiply here
    for (int tileElement = 0;
	 tileElement < F2x2_3x3INPUT_TILE_SIZE * F2x2_3x3INPUT_TILE_SIZE;
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
		     m += F2x2_3x3INPUT_TILE_SIZE * F2x2_3x3INPUT_TILE_SIZE,
		     ++mTmp)
	    {
		*m = *mTmp;
	    }
	}
    }

    m = M;
    for (int kernel = 0;
	 kernel < kernels;
	 ++kernel)
    {
	for (int tile = 0;
	     tile < tiles;
	     ++tile,
		 output += F2x2_3x3OUTPUT_TILE_SIZE * F2x2_3x3OUTPUT_TILE_SIZE,
		 m += F2x2_3x3INPUT_TILE_SIZE * F2x2_3x3INPUT_TILE_SIZE)
	{
	    // Y[k,t] = A_T * m * A
	    // A: 4x2, m: 4x4: Y[k,t]: 2x2 (F(2x2, 3x3))

	    matmulSlow(F2x2_3x3A_T, m, outputTmp, F2x2_3x3OUTPUT_TILE_SIZE,
		       F2x2_3x3INPUT_TILE_SIZE, F2x2_3x3INPUT_TILE_SIZE);
	    
	    matmulSlow(outputTmp, F2x2_3x3A, output, F2x2_3x3OUTPUT_TILE_SIZE,
		       F2x2_3x3OUTPUT_TILE_SIZE, F2x2_3x3INPUT_TILE_SIZE);
	    	    
	}
    }
}
 
