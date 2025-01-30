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

void imageToColumns(float* patches, float* image, int imageSize,
		    int nChannels, int kernelSize, int kernelStride)
{
    int pxlsPerChannel = imageSize * imageSize;

    for (int channel = 0;
	 channel < nChannels;
	 ++channel)
    {
	for (int kernelRow = 0;
	     kernelRow < kernelSize;
	     ++kernelRow)
	{
	    for (int kernelCol = 0;
		 kernelCol < kernelSize;
		 ++kernelCol)
	    {
		for (int imageRow = 0;
		     imageRow < imageSize;
		     imageRow += kernelStride)
		{
		    for (int imageCol = 0;
			 imageCol < imageSize;
			 imageCol += kernelStride, ++patches)
		    {
			*patches = image[channel * pxlsPerChannel +
					 (imageRow + kernelRow) * imageSize +
					 (imageCol + kernelCol)];
		    }
		}
	    }
	}
    }
}

void imageToRows(float* patches, float* image, int imageSize,
		 int nChannels, int kernelSize, int kernelStride)
{
    int pxlsPerChannel = imageSize * imageSize;
    for (int imageRow = 0;
	 imageRow < imageSize;
	 imageRow += kernelStride)
    {
	for (int imageCol = 0;
	     imageCol < imageSize;
	     imageCol += kernelStride)
	{
	    for (int channel = 0;
		 channel < nChannels;
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
					 (imageRow + kernelRow) * imageSize +
					 (imageCol + kernelCol)];
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
    float patches[4 * 4 * 3] = {0};
    // imageToRows(patches, testImage, 4, 3, 2, 2);

    imageToColumns(patches, testImage, 4, 3, 2, 2);
    float conv[2 * 4] = {0};
    matmulSlow(testKernelRow, patches, conv, 2, 4, 2 * 2 * 3);
    // float conv[4 * 2] = {0}; 
    // matmulSlow(patches, testKernel, conv, 4, 2, 2 * 2 * 3);
    printf("");
}
