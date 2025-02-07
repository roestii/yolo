import torch
from f2x2_3x3_convolution_ref import f2x2_3x3_convolution

def convolution_backward(dloutput, input, kernel):
    # Let's suppose that the output is of shape 2 * 6 * 6, the input is of shape 3 * 8 * 8, and the
    # kernel is of shape 2 * 3 * 3 * 3


    # We can compute the dL/dkernel using the dL/doutput and the input and compute the convolution of
    # the input using the dL/doutput a tile at a time, where one none overlapped 2 * 2 output tile was
    # produced by a 4 * 4 overlapped input tile

    K = kernel.shape[0] # suppose this is 2
    C = kernel.shape[1] # suppose this is 3
    m = 2
    alpha = 4
    dlkernel = torch.zeros_like(kernel)

    # dL/dkernel
    for outputChannel in range(K):
        for inputChannel in range(C):
            # We now want to calculate dL/dkernel[outputChannel, inputChannel] using
            # dL/doutput[outputChannel] and input[inputChannel]
            # So in this case for the f3x3_2x2 convolution C and K are 1, thus we can simplify
            # the function down below

            # TODO(louis): We have to tile the output, how do we specify the tile size? 
            # (maybe hardcode it to 2?)

            # In this case the overlap is the same as the output tile size which means
            # that we can use tileRow and tileCol for both dloutput and input as an index.
            for tileRow in range(0, dloutput.shape[-1], m):
                for tileCol in range(0, dloutput.shape[-1], m):
                    dloutputTile = dloutput[outputChannel, tileRow:tileRow+m, tileCol:tileCol+m]
                    inputTile = input[inputChannel, tileRow:tileRow+alpha, tileCol:tileCol+alpha]
                    dlkernel[outputChannel, inputChannel] += simplified_f3x3_2x2_convolution(inputTile, dloutputTile)

    # dL/dinput
    # Calculating dL/dinput can be formulated as a convolution of a padded version of dL/doutput using
    # the kernel rotated by 180 degrees. Each input channel contributes to each output channel using
    # the corresponding channel of a kernel.
    # That is dL/dinput[channel] = sum(outputChannel, padded(dL/doutput[outputChannel]) * flipped(kernel[outputChannel, inputChannel))
    r = kernel.shape[-1]
    padding = 2 * (r - 1)
    dlinput = torch.zeros_like(input)
    paddeddloutput = torch.zeros(dloutput.shape[0], padding + dloutput.shape[-1], padding + dloutput.shape[-1])
    paddeddloutput[:, r-1:r-1 + dloutput.shape[-1], r-1:r-1 + dloutput.shape[-1]] = dloutput
    for inputChannel in range(C):
        for outputChannel in range(K):
            for tileRow in range(0, input.shape[-1], m):
                for tileCol in range(0, input.shape[-1], m):
                    dloutputTile = paddeddloutput[outputChannel, tileRow:tileRow+alpha, tileCol:tileCol+alpha]
                    flippedKernel = kernel[outputChannel, inputChannel].rot90().rot90()
                    breakpoint()
                    dlinput[inputChannel, tileRow:tileRow+m, tileCol:tileCol+m] += simplified_f2x2_3x3_convolution(dloutputTile, flippedKernel)

    return dlkernel, dlinput

# The input corresponds to a singular tile of size alpha
def simplified_f2x2_3x3_convolution(input, kernel):
    m = 2
    r = 3
    alpha = m + r - 1

    C = input.shape[0]
    K = kernel.shape[0]
    P = (input.shape[-1] - r + 1) ** 2 // 4
        
    B = torch.tensor([1.0, 0.0,  0.0,  0.0,
                      0.0, 1.0, -1.0,  1.0,
                      -1.0, 1.0,  1.0,  0.0,
                      0.0, 0.0,  0.0, -1.0]).reshape(4, 4)
    G = torch.tensor([1.0,  0.0, 0.0,
                      0.5,  0.5, 0.5,
                      0.5, -0.5, 0.5,
                      0.0,  0.0, 1.0]).reshape(4, 3)
    A = torch.tensor([1.0,  0.0,
                      1.0,  1.0,
                      1.0, -1.0,
                      0.0, -1.0]).reshape(4, 2)

    Y = A.T @ ((G @ kernel @ G.T) * (B.T @ input @ B)) @ A
    return Y

def simplified_f3x3_2x2_convolution(input, kernel):
    B = torch.tensor([ 1, 0,  0,  0,
                       0, 1, -1, -1,
                      -1, 1,  1,  0,
                       0, 0,  0,  1],
                     dtype=torch.float).reshape(4, 4)
    A = torch.tensor([1,  0, 0,
                      1,  1, 1,
                      1, -1, 1,
                      0,  0, 1],
                     dtype=torch.float).reshape(4, 3)
    G = torch.tensor([1.0,  0.0,
                      0.5,  0.5,
                      0.5, -0.5,
                      0.0,  1.0]).reshape(4, 2)

    Y = A.T @ ((G @ kernel @ G.T) * (B.T @ input @ B)) @ A
    return Y
    

def f3x3_2x2_convolution(input, kernel):
    m = 3
    r = 2
    alpha = m + r - 1

    C = input.shape[0]
    K = kernel.shape[0]
    P = (input.shape[-1] - r + 1) ** 2 // 4
        
    B = torch.tensor([1.0, 0.0,  0.0,  0.0,
                      0.0, 1.0, -1.0,  1.0,
                      -1.0, 1.0,  1.0,  0.0,
                      0.0, 0.0,  0.0, -1.0]).reshape(4, 4)
    G = torch.tensor([1.0,  0.0, 0.0,
                      0.5,  0.5, 0.5,
                      0.5, -0.5, 0.5,
                      0.0,  0.0, 1.0]).reshape(4, 3)
    A = torch.tensor([1.0,  0.0,
                      1.0,  1.0,
                      1.0, -1.0,
                      0.0, -1.0]).reshape(4, 2)
    
    U = torch.zeros(K, C, alpha, alpha)
    V = torch.zeros(C, P, alpha, alpha)
    M = torch.zeros(K, P, alpha, alpha)
    # TODO(louis): check the size of the C implementation here as well
    Y = torch.zeros(K, P, m, m)

    for k in range(K):
        for c in range(C):
            g = kernel[k, c]
            U[k, c] = G @ g @ G.T
    # TODO(louis): there is a bug here, how do we get the tile indices?
    for c in range(C):
        b = 0
        for x in range(0, input.shape[-1] - alpha + 1, m):
            for y in range(0, input.shape[-1] - alpha + 1, m):
                d = input[c, x:x+alpha, y:y+alpha]
                V[c, b] = B.T @ d @ B
                b += 1

    for eps in range(alpha):
        for ups in range(alpha):
            M[:, :, eps, ups] = U[:, :, eps, ups] @ V[:, :, eps, ups]

    for k in range(K):
        for b in range(P):
            m = M[k, b]
            Y[k, b] = A.T @ M[k, b] @ A

    return Y
