import numpy
from f2x2_3x3_convolution_ref import simplified_f2x2_3x3_convolution, f2x2_3x3_convolution
from f3x3_2x2_convolution_ref import simplified_f3x3_2x2_convolution

def convolution_forward(input, kernel):
    B = input.shape[0]
    outputChannels = kernel.shape[0]
    outputSize = input.shape[-1] - kernel.shape[-1] + 1
    output = numpy.zeros((B, outputChannels, outputSize, outputSize))
    for b in range(B):
        output[b] = f2x2_3x3_convolution(input[b], kernel)
        
    return output

def convolution_backward(dloutput, input, kernel):
    # Let's suppose that the output is of shape 2 * 6 * 6, the input is of shape 3 * 8 * 8, and the
    # kernel is of shape 2 * 3 * 3 * 3


    # We can compute the dL/dkernel using the dL/doutput and the input and compute the convolution of
    # the input using the dL/doutput a tile at a time, where one none overlapped 2 * 2 output tile was
    # produced by a 4 * 4 overlapped input tile

    B = input.shape[0]
    K = kernel.shape[0] # suppose this is 2
    C = kernel.shape[1] # suppose this is 3
    m = 2
    alpha = 4
    dlkernel = numpy.zeros_like(kernel)

    # dL/dkernel
    for b in range(B):
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
                        dloutputTile = dloutput[b, outputChannel, tileRow:tileRow+m, tileCol:tileCol+m]
                        inputTile = input[b, inputChannel, tileRow:tileRow+alpha, tileCol:tileCol+alpha]
                        dlkernel[outputChannel, inputChannel] += simplified_f3x3_2x2_convolution(inputTile, dloutputTile)

    # dL/dinput
    # Calculating dL/dinput can be formulated as a convolution of a padded version of dL/doutput using
    # the kernel rotated by 180 degrees. Each input channel contributes to each output channel using
    # the corresponding channel of a kernel.
    # That is dL/dinput[channel] = sum(outputChannel, padded(dL/doutput[outputChannel]) * flipped(kernel[outputChannel, inputChannel))
    r = kernel.shape[-1]
    padding = 2 * (r - 1)
    dlinput = numpy.zeros_like(input)
    paddeddloutput = numpy.zeros((B, dloutput.shape[1], padding + dloutput.shape[-1], padding + dloutput.shape[-1]))
    paddeddloutput[:, :, r-1:r-1 + dloutput.shape[-1], r-1:r-1 + dloutput.shape[-1]] = dloutput

    for b in range(B):
        for inputChannel in range(C):
            for outputChannel in range(K):
                flippedKernel = numpy.rot90(numpy.rot90(kernel[outputChannel, inputChannel]))
                for tileRow in range(0, input.shape[-1], m):
                    for tileCol in range(0, input.shape[-1], m):
                        dloutputTile = paddeddloutput[b, outputChannel, tileRow:tileRow+alpha, tileCol:tileCol+alpha]
                        dlinput[b, inputChannel, tileRow:tileRow+m, tileCol:tileCol+m] += simplified_f2x2_3x3_convolution(dloutputTile, flippedKernel)

    return dlkernel, dlinput
