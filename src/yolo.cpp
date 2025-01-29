#include "types.h"
#include "test_data.h"
#include <stdio.h>

void matmulSlow(float* a, float* b, float* c, int m, int n, int k)
{
    // NOTE(louis): a is m x k, b is k x n, thus c has to be m x n
    for (int i = 0; i < m; ++i)
    {
	for (int j = 0; j < n; ++j, ++c)
	{
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
	    for (int l = 0; l < k; ++l)
	    {
		*c += a[i * k + l] * b[j * k + l];
	    }
	}
    }
}

void imageToRows(float* patches, float* image, int imageSize,
		 int nChannels, int kernelSize, int kernelStride,
		 int nPatches)
{
    int pxlsPerChannel = imageSize * imageSize;
    for (int imageRow = 0;
	 row < imageSize;
	 row += kernelStride)
    {
	for (int imageCol = 0;
	     col < imageSize;
	     col += kernelStride)
	{
	    for (int channel = 0;
		 channel = nChannels;
		 ++channel)
	    {
		for (int kernelRow = 0;
		     kernelRow < kernelSize;
		     ++kernelRow)
		{
		    for (int kernelCol = 0;
			 kernelCol < kernelSize;
			 ++kernelCol, ++patches)
		    {
			*patches = image[channel * pxlsPerChannel +
				      (row + kernelRow) * imageSize +
				      (col + kernelCol)];
		    }
		}
	    }
	}
    }
}

void columnsToImage(float* image, float* cols, int )
{
}

int main()
{
    float a[] = { 1, 2, 3, 4 };
    float b[] = { 5, 6, 7, 8 };
    float c[4] = { 0 };
    float d[4] = { 0 };
    float e[4] = { 0 };

    matmulSlow(a, b, c, 2, 2, 2);
    printf("a * b\n");
    for (int i = 0; i < 4; ++i)
    {
	printf("%f\n", c[i]);
    }

    matmulATransposedB(a, b, d, 2, 2, 2);
    printf("a^T * b\n");
    for (int i = 0; i < 4; ++i)
    {
	printf("%f\n", d[i]);
    }
    
    matmulABTransposed(a, b, e, 2, 2, 2);
    printf("a * b^T\n");
    for (int i = 0; i < 4; ++i)
    {
	printf("%f\n", e[i]);
    }

    
    return 0;
}
