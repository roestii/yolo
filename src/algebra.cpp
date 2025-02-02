#include "algebra.h"

void matmulSlow(float* a, float* b, float* c, int m, int n, int k)
{
    // NOTE(louis): a is m x k, b is k x n, thus c has to be m x n
    for (int i = 0; i < m; ++i)
    {
	for (int j = 0; j < n; ++j, ++c)
	{
	    *c = 0;
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
	    *c = 0;
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
	    *c = 0;
	    for (int l = 0; l < k; ++l)
	    {
		*c += a[i * k + l] * b[j * k + l];
	    }
	}
    }
}


