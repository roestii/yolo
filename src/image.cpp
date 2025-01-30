#include "image.h"

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
		     imageRow < imageSize - kernelSize + 1;
		     imageRow += kernelStride)
		{
		    for (int imageCol = 0;
			 imageCol < imageSize - kernelSize + 1;
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
	 imageRow < imageSize - kernelSize + 1;
	 imageRow += kernelStride)
    {
	for (int imageCol = 0;
	     imageCol < imageSize - kernelSize + 1;
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
