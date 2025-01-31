#ifndef ALGEBRA_H
#define ALGEBRA_H

#include "types.h"

#define W3x3_OUTPUT_TILE_SIZE 2
#define W3x3_FILTER_SIZE 3
#define W3x3_INPUT_TILE_SIZE (W3x3_OUTPUT_TILE_SIZE + W3x3_FILTER_SIZE - 1) 

void matmulSlow(float*, float*, float*, int, int, int);
void matmulATransposedB(float*, float*, float*, int, int, int);
void matmulABTransposed(float*, float*, float*, int, int, int);

void winograd3x3(float*, float*, float*, float*,
		 float*, float*, float*, float*,
		 int, int, int, int);

#endif
