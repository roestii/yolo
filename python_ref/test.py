import numpy
from convolution import convolution_forward, convolution_backward

EPS = 1e-3

def load(name):
    with open(name, "rb") as f:
        return numpy.load(f)

testMiniBatch = load("../data/testMiniBatch.npy")
testKernel = load("../data/testKernel.npy")
testOutput = load("../data/testOutput.npy")
dlTestMiniBatch = load("../data/dlTestMiniBatch.npy")
dlTestKernel = load("../data/dlTestKernel.npy")
dlTestOutput = load("../data/dlTestOutput.npy")
breakpoint()

def test_convolution_backward():
    dlkernel, dlinput = convolution_backward(dlTestOutput, testMiniBatch, testKernel)
    
def test_convolution_forward():
    output = convolution_forward(testMiniBatch, testKernel)
   
if __name__ == "__main__":
    test_convolution_backward()
    test_convolution_forward()
