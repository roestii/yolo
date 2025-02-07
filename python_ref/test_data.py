import torch
import numpy

BATCH_SIZE = 32
KERNEL_SIZE = 3
INPUT_CHANNELS = 3
INPUT_SIZE = 112
OUTPUT_SIZE = INPUT_SIZE - KERNEL_SIZE + 1
OUTPUT_CHANNELS = 4

def save(name, arr):
    with open(name, "wb") as f:
        numpy.save(f, arr.numpy(), allow_pickle=False)

def main():
    testMiniBatch = torch.randn(BATCH_SIZE, INPUT_CHANNELS,
                                INPUT_SIZE, INPUT_SIZE,
                                requires_grad=True)
    testKernel = torch.randn(OUTPUT_CHANNELS, INPUT_CHANNELS,
                             KERNEL_SIZE, KERNEL_SIZE,
                             requires_grad=True)

    testMiniBatch.retain_grad()
    testKernel.retain_grad()
    testOutput = torch.nn.functional.conv2d(testMiniBatch, testKernel, stride=1)
    testOutput.retain_grad()
    
    expected = torch.randn_like(testOutput)
    loss = (expected - testOutput).square().sum()
    loss.backward()

    dlTestOutput = testOutput.grad.clone()
    dlTestMiniBatch = testMiniBatch.grad.clone()
    dlTestKernel = testKernel.grad.clone()

    with torch.no_grad():
        save("data/testMiniBatch.npy", testMiniBatch)
        save("data/testKernel.npy", testKernel)
        save("data/testOutput.npy", testOutput)
        save("data/dlTestOutput.npy", dlTestOutput)
        save("data/dlTestMiniBatch.npy", dlTestMiniBatch)
        save("data/dlTestKernel.npy", dlTestKernel)

if __name__ == "__main__":
    main()
