import numpy

def simplified_f3x3_2x2_convolution(input, kernel):
    B = numpy.array([ 1, 0,  0,  0,
                      0, 1, -1, -1,
                      -1, 1,  1,  0,
                      0, 0,  0,  1],
                     dtype=numpy.float32).reshape(4, 4)
    A = numpy.array([1,  0, 0,
                     1,  1, 1,
                     1, -1, 1,
                     0,  0, 1],
                     dtype=numpy.float32).reshape(4, 3)
    G = numpy.array([1.0,  0.0,
                     0.5,  0.5,
                     0.5, -0.5,
                     0.0,  1.0]).reshape(4, 2)

    Y = A.T @ ((G @ kernel @ G.T) * (B.T @ input @ B)) @ A
    return Y
