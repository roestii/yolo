#ifndef F2x2_3x3_CONVOLUTION_H
#define F2x2_3x3_CONVOLUTION_H

#include "algebra.h"

#define F2x2_3x3OUTPUT_TILE_SIZE 2
#define F2x2_3x3FILTER_SIZE 3
#define F2x2_3x3INPUT_TILE_SIZE (F2x2_3x3OUTPUT_TILE_SIZE + F2x2_3x3FILTER_SIZE - 1)
#define F2x2_3x3TILE_OVERLAP F2x2_3x3FILTER_SIZE - 1

void f2x2_3x3Convolution(float*, float*, float*, float*,
			 float*, float*, float*, float*, float*,
			 int, int, int, int);
#endif
