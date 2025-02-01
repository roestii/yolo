import torch

def f2x2_3x3_convolution(input, kernel):
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

    breakpoint()

    for eps in range(alpha):
        for ups in range(alpha):
            M[:, :, eps, ups] = U[:, :, eps, ups] @ V[:, :, eps, ups]

    for k in range(K):
        for b in range(P):
            m = M[k, b]
            Y[k, b] = A.T @ M[k, b] @ A

    return Y
