#include "f3x3_2x2_convolution.h"

// According to Lavin and Gray, the matrics B^T, A^T, and G
static float B[] = {
     1, 0,  0,  0,
     0, 1, -1, -1,
    -1, 1,  1,  0,
     0, 0,  0,  1
};

static float B_T[] = {
    1,  0, -1,  0,
    0,  1,  1,  0,
    0, -1,  1,  0,
    0, -1,  0,  1
};

static float G[] = {
    1.0,  0.0,
    0.5,  0.5,
    0.5, -0.5,
    0.0,  1.0
};

static float G_T[] = {
    1.0, 0.5,  0.5, 0.0,
    0.0, 0.5, -0.5, 1.0,
};

static float A[] = {
    1,  0, 0,
    1,  1, 1,
    1, -1, 1,
    0,  0, 1
};

static float A_T[] = {
    1, 1,  1,  0,
    0, 1, -1,  0,
    0, 1,  1,  1
};

static float Gg[F3x3_2x2INPUT_TILE_SIZE * F3x3_2x2FILTER_SIZE];
static float B_Td[F3x3_2x2INPUT_TILE_SIZE * F3x3_2x2INPUT_TILE_SIZE];
static float GgG_T[F3x3_2x2INPUT_TILE_SIZE * F3x3_2x2INPUT_TILE_SIZE];
static float B_TdB[F3x3_2x2INPUT_TILE_SIZE * F3x3_2x2INPUT_TILE_SIZE];
static float A_TM[F3x3_2x2OUTPUT_TILE_SIZE * F3x3_2x2INPUT_TILE_SIZE];

void f3x3_2x2SingleTileConvolution(float* Y, float* inputTile, float* kernel)
{
    // This method is used for the backward pass where we perform a convolution with
    // a single input tile that has one channel and a single kernel with one channel as well.
    // The output is given by Y = A^T ((G g G^T) * (B.T d B)) A
    matmulSlow(G, kernel, Gg, F3x3_2x2INPUT_TILE_SIZE,
	       F3x3_2x2FILTER_SIZE, F3x3_2x2FILTER_SIZE);
    matmulSlow(Gg, G_T, GgG_T, F3x3_2x2INPUT_TILE_SIZE,
	       F3x3_2x2INPUT_TILE_SIZE, F3x3_2x2FILTER_SIZE);
    matmulSlow(B_T, inputTile, B_Td, F3x3_2x2INPUT_TILE_SIZE,
	       F3x3_2x2INPUT_TILE_SIZE, F3x3_2x2INPUT_TILE_SIZE);
    matmulSlow(B_Td, B, B_TdB, F3x3_2x2INPUT_TILE_SIZE,
	       F3x3_2x2INPUT_TILE_SIZE, F3x3_2x2INPUT_TILE_SIZE);

    // TODO(louis): simd for this element wise multiply
    float* m = GgG_T;
    float* b = B_TdB;
    for (int i = 0;
	 i < F3x3_2x2INPUT_TILE_SIZE * F3x3_2x2INPUT_TILE_SIZE;
	 ++i, ++m, ++b)
    {
	*m *= *b;
    }
    
    m = GgG_T;
    matmulSlow(A_T, m, A_TM, F3x3_2x2OUTPUT_TILE_SIZE,
	       F3x3_2x2INPUT_TILE_SIZE, F3x3_2x2INPUT_TILE_SIZE);
    matmulSlow(A_TM, A, Y, F3x3_2x2OUTPUT_TILE_SIZE,
	       F3x3_2x2OUTPUT_TILE_SIZE, F3x3_2x2INPUT_TILE_SIZE);
}
