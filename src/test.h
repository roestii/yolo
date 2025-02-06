#ifndef TEST_DATA_H
#define TEST_DATA_H

#include "f2x2_3x3_convolution.h"
#include "layers.h"
#include <math.h>

#define TEST_FILTER_CHANNELS 3
#define TEST_FILTER_SIZE 3
#define TEST_IMAGE_SIZE 8
#define TEST_IMAGE_CHANNELS 3
#define TEST_OUTPUT_SIZE 6
#define TEST_TILES TEST_OUTPUT_SIZE * TEST_OUTPUT_SIZE / 4
#define EPS 1e-2

// 3 * 8 * 8
static float testImage[] = {
    1.0030e+00, -7.9719e-01, -1.2836e+00,  1.2146e+00,  1.2180e+00, -1.2004e-01, -3.0324e-01,  5.7025e-01,
    -1.5090e+00,  1.8884e+00, -8.3879e-01, -4.1456e-01,  1.0871e+00, -2.3732e-01,  1.5855e-01, -2.5492e+00,
    2.8172e-01, -1.0152e+00,  8.1117e-01,  1.0030e+00, 4.6480e-01,  1.4884e+00,  9.8615e-01,  5.0825e-01,
    -1.6607e+00, 2.4470e-01,  7.8692e-01, -1.4318e-01, -7.1970e-02,  1.2011e+00, 1.6911e+00,  2.4500e-02,
    2.2925e-01, -2.2687e-01,  9.2714e-02, 9.4118e-01, -2.1650e-01,  1.4567e-01, -2.0871e-01, -3.5467e-01,
    -1.0009e-02, -3.7680e-01,  2.4436e-01, -8.9139e-01,  1.5121e+00,
    -5.4045e-01, -1.7853e-01,  1.5121e-02,  8.5428e-01, -4.9648e-01,
    8.6770e-01, -3.3097e-01,  8.0293e-01, -1.6081e+00,  1.5604e+00,
    -1.0236e+00,  1.4416e+00,  1.3520e-01, -3.0394e-01,  5.7589e-01,
    -7.6641e-01,  7.6427e-02,  2.3247e-02, -1.0174e+00,  7.7632e-02,
    3.4804e-01, -1.5159e-01,  8.5462e-03,  1.2346e+00, -6.5222e-01,
    6.0301e-01, -2.0244e-01, -2.0114e+00,  1.2186e+00,  3.8402e-01,
    -1.1989e-01, -1.7145e-01,  2.2940e+00,  2.1418e-01, -6.9910e-01,
    6.5492e-01, -1.1970e+00,  4.4018e-01, -1.9285e-01,  4.5225e-01,
    -1.9642e-01, -6.9077e-01,  2.2332e-01,  7.4665e-01, -5.1299e-01,
    -1.8662e-01,  4.4141e-01,  6.8881e-01, -3.3760e-02,  3.3350e-01,
    -4.4507e-01,  1.6321e+00, -1.1717e+00, -7.3580e-01, -2.9832e-01,
    3.8535e-01, -8.2510e-02, -1.3809e+00, -2.8915e-01, -1.4925e+00,
    -4.6710e-01, -1.0300e+00,  1.8371e-01, -5.1687e-01,  6.1722e-01,
    7.0307e-01,  1.1222e+00,  1.9273e+00, -8.7348e-01, -1.0212e+00,
    7.8699e-02, -1.9553e-01,  1.9319e+00, -4.5184e-01,  4.7122e-03,
    -4.9319e-01,  1.1754e+00, -8.7940e-02, -8.5939e-01, -1.5832e+00,
    4.4375e-01, -3.6743e-02, -6.2386e-01, -1.9656e-03,  4.6986e-01,
    -1.3896e+00, -5.4676e-01, -6.8569e-01, -1.3891e+00, -3.1386e-01,
    1.0981e+00,  5.2239e-01,  2.2071e-01, -1.5734e+00,  5.4456e-02,
    7.3507e-02,  7.7122e-01, -2.0417e-01,  8.7524e-01,  9.2527e-01,
    -1.2807e+00, -1.6179e+00,  7.0662e-01, -1.6992e-01, -4.0098e-01,
    -8.6504e-01, -1.4605e+00, -1.4541e-01,  9.6541e-01, -2.0080e-01,
    -6.3385e-01,  1.4063e+00, -4.9919e-01, -1.3140e+00,  1.3475e-01,
    2.4074e-02,  1.5055e+00,  2.7961e-04, -5.4913e-01,  3.5659e-01,
    1.3284e+00,  1.7589e-01, -1.7908e-02,  7.3478e-01,  6.9558e-01,
    1.0634e+00, -2.4920e-01, -2.3577e-02, -1.6739e-01, -8.4925e-01,
    -6.7817e-01,  1.4261e+00,  9.0194e-01, -7.4779e-01,  3.4431e-02,
    -2.0981e-01,  1.5640e+00, -8.7516e-02, -6.3122e-01, -7.4941e-01,
    1.1275e+00, -4.2285e-01, -9.1008e-01,  9.2318e-02, -1.2455e+00,
    -8.3147e-01, -1.2037e+00
};

// 3 * 3 * 3 * 3
static float testKernel[] = {
     0.1822, -0.6083,  1.0995, -0.1266, -0.6512, -0.4662,  1.3823, -1.2718,
     0.0973, -0.4191,  0.7098, -0.3835, -0.0081,  1.8420, -0.0353,  0.2822,
     0.4658, -0.1466, -0.9037, -1.1712,  0.7681, -0.6676, -1.4929, -0.6909,
     -0.1331,  0.4322,  0.3646,  0.6162,  1.4353,  1.2334, -0.5924, -1.5537,
     -0.8755, -1.5511, -0.7107,  0.7584,  0.6705,  1.2914,  0.7411, -0.1571,
     -0.7939,  1.1706, -0.7477, -0.0447,  0.5450,  0.8718,  1.2464,  0.0212,
     -0.2375, -0.1679,  1.1624, -0.0704, -1.0220,  2.0083,  0.4414, -1.1360,
     -0.2243,  1.3167,  0.0421,  1.6798, -1.2842,  0.8175,  1.3141,  0.6325,
     -0.7352,  0.1596, -0.4667, -0.2783,  0.2414,  1.1603, -1.0823, -2.7082,
     0.0552, -0.1388,  0.7002, -0.6510,  0.9268, -0.1423,  2.8307, -0.0581,
     -1.8185
};

void testConvolutionForward()
{
    float output[TEST_FILTER_CHANNELS * TEST_TILES *
			F2x2_3x3OUTPUT_TILE_SIZE * F2x2_3x3OUTPUT_TILE_SIZE] = {0};
    float U[TEST_FILTER_CHANNELS * TEST_IMAGE_CHANNELS *
		   F2x2_3x3INPUT_TILE_SIZE * F2x2_3x3INPUT_TILE_SIZE] = {0};
    float V[TEST_IMAGE_CHANNELS * TEST_TILES *
		   F2x2_3x3INPUT_TILE_SIZE * F2x2_3x3INPUT_TILE_SIZE] = {0};
    float M[TEST_FILTER_CHANNELS * TEST_TILES *
		   F2x2_3x3INPUT_TILE_SIZE * F2x2_3x3INPUT_TILE_SIZE] = {0};
    float Utmp[TEST_FILTER_CHANNELS * TEST_IMAGE_CHANNELS] = {0};
    float Vtmp[TEST_IMAGE_CHANNELS * TEST_TILES] = {0};
    float Mtmp[TEST_FILTER_CHANNELS * TEST_TILES] = {0};

    float expected[] = {
	-0.0227,  3.1503,  2.2196, -0.1066,  2.3752,  4.3258,
	-4.4585,  3.2144,  2.9045, -0.7439, -1.7257, -3.7768,
	-1.4922,  2.7987, -0.6163,  3.1156, -0.4098,  1.2584,
	-5.0871, -5.5150,  3.0065, -3.7423, -1.0198, -2.4282,
	-3.5687, -7.2243,  1.4735, -0.7978,  5.3112, -3.6643,
	-0.3914, -5.2213,  1.0831, -2.7433, -0.2249, -1.6548,
	-5.9655,  3.7522, -0.7972,  3.1586, -7.0933, -3.8931,
	3.1125, -3.1726, -0.0541, -2.8078, -2.5397, -6.4351,
	-5.4950, -2.6114,  5.5279,  2.9568, -4.2548, -3.6926,
	3.1775, -1.3929,  2.2072,  4.3760,  0.4891,  2.7722,
	-3.2115,  3.0090, -1.2917,  3.3826, -0.3870, -2.7874,
	-7.9844, -2.7095, -0.5225,  3.3562,  0.8631, -4.2981,
	4.2045, -2.4310, -5.4525,  0.5528,  9.5501, -5.6429,
	-0.3768,  5.0723, -3.1724,  3.3745, 10.2058,  4.1083,
	6.2456,  4.8398, -2.1411, -3.6407,  4.4391,  4.6554,
	2.2588,  1.7787,  5.0306,  1.2947, -6.4834, -3.9510,
	13.8102,  1.4928, -1.4367, -9.3480, -0.1981, 10.1397,
	-3.1993,  7.2083,  7.0802, -6.6493,  5.4392, -6.3684
    };
    
    f2x2_3x3Convolution(output, U, V, M, Utmp, Vtmp, Mtmp,
			testImage, testKernel, TEST_IMAGE_SIZE,
			TEST_IMAGE_CHANNELS, TEST_FILTER_CHANNELS,
			TEST_TILES);
    for (int i = 0;
	 i < sizeof(output) / sizeof(float);
	 ++i)
    {
	assert(abs(output[i] - expected[i]) < EPS);
    }
}

#define PADDED_SIZE (TEST_OUTPUT_SIZE + 2 * PADDING)

void testConvolutionBackward()
{
    static float input[TEST_INPUT_CHANNELS * TEST_INPUT_SIZE * TEST_INPUT_SIZE] = {
    };
    static float kernel[TEST_OUTPUT_CHANNELS * F2x2_3x3FILTER_SIZE * F2x2_3x3FILTER_SIZE] = {
    };
    static float dloutput[TEST_OUTPUT_CHANNELS * TEST_OUTPUT_SIZE * TEST_OUTPUT_SIZE] = {
    };
    static float paddeddloutput[TEST_OUTPUT_CHANNELS * PADDED_SIZE * PADDED_SIZE] = {0};
    static float dlinput[sizeof(input) / sizeof(float)] = {
    };
    static float dlkernel[sizeof(kernel) / sizeof(float)] = {
    };
    static float dlinputExpected[sizeof(input) / sizeof(float)] = {
    };
    static float dlkernelExpected[sizeof(kernel) / sizeof(float)] = {
    };

    convolutionBackward(dlinput, dlkernel, dloutput, paddeddloutput, kernel, input,
			TEST_INPUT_CHANNELS, TEST_INPUT_SIZE, TEST_OUTPUT_CHANNELS, TEST_OUTPUT_SIZE);
    for (int i = 0;
	 i < sizeof(dlkernel) / sizeof(float);
	 ++i)
    {
	assert(abs(dlkernel[i] - dlkernelExpected[i]) < EPS);
    }

    for (int i = 0;
	 i < sizeof(dlinput) / sizeof(float);
	 ++i)
    {
	assert(abs(dlinput[i] - dlinputExpected[i]) < EPS);
    }
}

/* void testF3x3_2x2Convolution()
{
    float output[TEST_FILTER_CHANNELS * TEST_TILES *
			F3x3_2x2OUTPUT_TILE_SIZE * F3x3_2x2OUTPUT_TILE_SIZE] = {0};
    float U[TEST_FILTER_CHANNELS * TEST_IMAGE_CHANNELS *
		   F3x3_2x2INPUT_TILE_SIZE * F3x3_2x2INPUT_TILE_SIZE] = {0};
    float V[TEST_IMAGE_CHANNELS * TEST_TILES *
		   F3x3_2x2INPUT_TILE_SIZE * F3x3_2x2INPUT_TILE_SIZE] = {0};
    float M[TEST_FILTER_CHANNELS * TEST_TILES *
		   F3x3_2x2INPUT_TILE_SIZE * F3x3_2x2INPUT_TILE_SIZE] = {0};
    float Utmp[TEST_FILTER_CHANNELS * TEST_IMAGE_CHANNELS] = {0};
    float Vtmp[TEST_IMAGE_CHANNELS * TEST_TILES] = {0};
    float Mtmp[TEST_FILTER_CHANNELS * TEST_TILES] = {0};

    float expected[sizeof(output) / sizeof(float)] = {
	-0.0227,  3.1503,  2.2196, -0.1066,  2.3752,  4.3258,
	-4.4585,  3.2144,  2.9045, -0.7439, -1.7257, -3.7768,
	-1.4922,  2.7987, -0.6163,  3.1156, -0.4098,  1.2584,
	-5.0871, -5.5150,  3.0065, -3.7423, -1.0198, -2.4282,
	-3.5687, -7.2243,  1.4735, -0.7978,  5.3112, -3.6643,
	-0.3914, -5.2213,  1.0831, -2.7433, -0.2249, -1.6548,
	-5.9655,  3.7522, -0.7972,  3.1586, -7.0933, -3.8931,
	3.1125, -3.1726, -0.0541, -2.8078, -2.5397, -6.4351,
	-5.4950, -2.6114,  5.5279,  2.9568, -4.2548, -3.6926,
	3.1775, -1.3929,  2.2072,  4.3760,  0.4891,  2.7722,
	-3.2115,  3.0090, -1.2917,  3.3826, -0.3870, -2.7874,
	-7.9844, -2.7095, -0.5225,  3.3562,  0.8631, -4.2981,
	4.2045, -2.4310, -5.4525,  0.5528,  9.5501, -5.6429,
	-0.3768,  5.0723, -3.1724,  3.3745, 10.2058,  4.1083,
	6.2456,  4.8398, -2.1411, -3.6407,  4.4391,  4.6554,
	2.2588,  1.7787,  5.0306,  1.2947, -6.4834, -3.9510,
	13.8102,  1.4928, -1.4367, -9.3480, -0.1981, 10.1397,
	-3.1993,  7.2083,  7.0802, -6.6493,  5.4392, -6.3684
    };
    
    f3x3_2x2Convolution(output, U, V, M, Utmp, Vtmp, Mtmp,
			testImage, testKernel, TEST_IMAGE_SIZE,
			TEST_IMAGE_CHANNELS, TEST_FILTER_CHANNELS,
			TEST_TILES);
    for (int i = 0;
	 i < sizeof(output) / sizeof(float);
	 ++i)
    {
	assert(abs(output[i] - expected[i]) < EPS);
    }
}
*/

#endif
