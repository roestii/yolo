#ifndef F3x3_2x2_CONVOLUTION_H
#define F3x3_2x2_CONVOLUTION_H

#include "algebra.h"

#define F3x3_2x2OUTPUT_TILE_SIZE 3
#define F3x3_2x2FILTER_SIZE 2
#define F3x3_2x2INPUT_TILE_SIZE (F3x3_2x2OUTPUT_TILE_SIZE + F3x3_2x2FILTER_SIZE - 1)
#define F3x3_2x2TILE_OVERLAP (F3x3_2x2FILTER_SIZE - 1)

void f3x3_2x2SingleTileConvolution(float*, float*, float*);
/* void f3x3_2x2Convolution(float*, float*, float*, float*,
			 float*, float*, float*, float*, float*,
			 int, int, int, int); */
#endif
