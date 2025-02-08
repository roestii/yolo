#include "numpy_parser.h"
#include "layers.h"
#include "arena.h"

#include <sys/mman.h>
#include <stdio.h>

#define BATCH_SIZE 32
#define KERNEL_SIZE 3
#define INPUT_CHANNELS 3
#define INPUT_SIZE 112
#define OUTPUT_SIZE (INPUT_SIZE - KERNEL_SIZE + 1)
#define OUTPUT_CHANNELS 4

#define TEST_KERNEL_SIZE OUTPUT_CHANNELS * INPUT_CHANNELS * F2x2_3x3FILTER_SIZE * F2x2_3x3FILTER_SIZE * sizeof(float)
#define TEST_MINI_BATCH_SIZE BATCH_SIZE * INPUT_CHANNELS * INPUT_SIZE * INPUT_SIZE * sizeof(float)
#define TEST_OUTPUT_SIZE BATCH_SIZE * OUTPUT_CHANNELS * OUTPUT_SIZE * OUTPUT_SIZE * sizeof(float)

#define TILES OUTPUT_SIZE * OUTPUT_SIZE / (F2x2_3x3OUTPUT_TILE_SIZE * F2x2_3x3OUTPUT_TILE_SIZE)
#define U_SIZE OUTPUT_CHANNELS * INPUT_CHANNELS * F2x2_3x3FILTER_SIZE * F2x2_3x3FILTER_SIZE
#define V_SIZE INPUT_CHANNELS * TILES * F2x2_3x3INPUT_TILE_SIZE * F2x2_3x3INPUT_TILE_SIZE
#define M_SIZE OUTPUT_CHANNELS * TILES * F2x2_3x3INPUT_TILE_SIZE * F2x2_3x3INPUT_TILE_SIZE
#define UTMP_SIZE OUTPUT_CHANNELS * INPUT_CHANNELS
#define VTMP_SIZE INPUT_CHANNELS * TILES
#define MTMP_SIZE OUTPUT_CHANNELS * TILES

// TODO(louis): We need additional memory here and we might want to page align this
#define MEMORY 3 * (TEST_KERNEL_SIZE + TEST_MINI_BATCH_SIZE + TEST_OUTPUT_SIZE) \
    + U_SIZE + V_SIZE + M_SIZE + UTMP_SIZE + VTMP_SIZE + MTMP_SIZE

/*int testConvolutionForward(float* input, float* output, )
{
    convolutionForward();
    for (int i = 0;
	 i < sizeof(output) / sizeof(float);
	 ++i)
    {
	assert(abs(output[i] - expected[i]) < EPS);
    }
}
*/

int testConvolutionBackward()
{
    // convolutionBackward();
    return 0;
}

int main()
{
    void* start = mmap(NULL, MEMORY, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (start == MAP_FAILED)
    {
	fprintf(stderr, "Cannot map memory.\n");
	return -1;
    }
    
    arena a;
    initArena(&a, (uintptr) start, MEMORY);
    float* testKernel = (float*) pushSize(&a, TEST_KERNEL_SIZE);
    float* testMiniBatch = (float*) pushSize(&a, TEST_MINI_BATCH_SIZE);
    float* testOutput = (float*) pushSize(&a, TEST_OUTPUT_SIZE);
    float* dlTestKernel = (float*) pushSize(&a, TEST_KERNEL_SIZE);
    float* dlTestMiniBatch = (float*) pushSize(&a, TEST_MINI_BATCH_SIZE);
    float* dlTestOutput = (float*) pushSize(&a, TEST_OUTPUT_SIZE);

    float* computedOutput = (float*) pushSize(&a, TEST_OUTPUT_SIZE);
    float* computedDlMiniBatch = (float*) pushSize(&a, TEST_MINI_BATCH_SIZE);
    float* computedDlKernel = (float*) pushSize(&a, TEST_KERNEL_SIZE);

    float* U = (float*) pushSize(&a, U_SIZE);
    float* V = (float*) pushSize(&a, V_SIZE);
    float* M = (float*) pushSize(&a, M_SIZE);
    float* Utmp = (float*) pushSize(&a, UTMP_SIZE);
    float* Vtmp = (float*) pushSize(&a, VTMP_SIZE);
    float* Mtmp = (float*) pushSize(&a, MTMP_SIZE);

    assert(load((char*) "data/testKernel.npy", testKernel, TEST_KERNEL_SIZE) == 0);
    assert(load((char*) "data/testMiniBatch.npy", testMiniBatch, TEST_MINI_BATCH_SIZE) == 0);
    assert(load((char*) "data/testOutput.npy", testOutput, TEST_OUTPUT_SIZE) == 0);
    assert(load((char*) "data/dlTestKernel.npy", dlTestKernel, TEST_KERNEL_SIZE) == 0);
    assert(load((char*) "data/dlTestMiniBatch.npy", dlTestMiniBatch, TEST_MINI_BATCH_SIZE) == 0);
    assert(load((char*) "data/dlTestOutput.npy", dlTestOutput, TEST_OUTPUT_SIZE) == 0);
       
    if (munmap(start, MEMORY) == -1)
    {
	fprintf(stderr, "Munmap failed.\n");
	return -1;
    }
}
