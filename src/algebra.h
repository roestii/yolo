#ifndef ALGEBRA_H
#define ALGEBRA_H

#include "types.h"

#define F2x2_3x3OUTPUT_TILE_SIZE 2
#define F2x2_3x3FILTER_SIZE 3
#define F2x2_3x3INPUT_TILE_SIZE (F2x2_3x3OUTPUT_TILE_SIZE + F2x2_3x3FILTER_SIZE - 1) 

void matmulSlow(float*, float*, float*, int, int, int);
void matmulATransposedB(float*, float*, float*, int, int, int);
void matmulABTransposed(float*, float*, float*, int, int, int);

void f2x2_3x3Convolution(float*, float*, float*, float*,
			 float*, float*, float*, float*,
			 int, int, int, int);

#endif
