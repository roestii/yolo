import numpy

# The input corresponds to a singular tile of size alpha
def simplified_f2x2_3x3_convolution(input, kernel):
    m = 2
    r = 3
    alpha = m + r - 1

    C = input.shape[0]
    K = kernel.shape[0]
    P = (input.shape[-1] - r + 1) ** 2 // 4
        
    B = numpy.array([1.0, 0.0,  0.0,  0.0,
                      0.0, 1.0, -1.0,  1.0,
                      -1.0, 1.0,  1.0,  0.0,
                      0.0, 0.0,  0.0, -1.0]).reshape(4, 4)
    G = numpy.array([1.0,  0.0, 0.0,
                      0.5,  0.5, 0.5,
                      0.5, -0.5, 0.5,
                      0.0,  0.0, 1.0]).reshape(4, 3)
    A = numpy.array([1.0,  0.0,
                     1.0,  1.0,
                     1.0, -1.0,
                     0.0, -1.0]).reshape(4, 2)

    Y = A.T @ ((G @ kernel @ G.T) * (B.T @ input @ B)) @ A
    return Y


def f2x2_3x3_convolution(input, kernel):
    m = 2
    r = 3
    alpha = m + r - 1

    C = input.shape[0]
    K = kernel.shape[0]
    P = (input.shape[-1] - r + 1) ** 2 // 4
        
    B = numpy.array([1.0, 0.0,  0.0,  0.0,
                     0.0, 1.0, -1.0,  1.0,
                     -1.0, 1.0,  1.0,  0.0,
                     0.0, 0.0,  0.0, -1.0]).reshape(4, 4)
    G = numpy.array([1.0,  0.0, 0.0,
                     0.5,  0.5, 0.5,
                     0.5, -0.5, 0.5,
                     0.0,  0.0, 1.0]).reshape(4, 3)
    A = numpy.array([1.0,  0.0,
                     1.0,  1.0,
                     1.0, -1.0,
                     0.0, -1.0]).reshape(4, 2)
    
    U = numpy.zeros((K, C, alpha, alpha))
    V = numpy.zeros((C, P, alpha, alpha))
    M = numpy.zeros((K, P, alpha, alpha))
    outputSize = input.shape[-1] - r + 1
    Y = numpy.zeros((K, outputSize, outputSize), dtype=numpy.float32)

    for k in range(K):
        for c in range(C):
            g = kernel[k, c]
            U[k, c] = G @ g @ G.T
   
    for c in range(C):
        b = 0
        for row in range(0, input.shape[-1] - alpha + 1, m):
            for col in range(0, input.shape[-1] - alpha + 1, m):
                d = input[c, row:row+alpha, col:col+alpha]
                V[c, b] = B.T @ d @ B
                b += 1

    for eps in range(alpha):
        for ups in range(alpha):
            M[:, :, eps, ups] = U[:, :, eps, ups] @ V[:, :, eps, ups]

    for k in range(K):
        b = 0
        for row in range(0, input.shape[-1] - alpha + 1, m):
            for col in range(0, input.shape[-1] - alpha + 1, m):
                Y[k, row:row+m, col:col+m] = A.T @ M[k, b] @ A
                b += 1

    return Y
