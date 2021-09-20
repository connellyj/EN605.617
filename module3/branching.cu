// Based on the work of Andrew Krepps
// Modifications by Julia Connelly

#include <chrono>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>

__global__
void branch(int* a, int* b, int* result)
{
	const unsigned int t = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (t % 4 == 0)
	{
		result[t] = a[t] + b[t];
	}
	else if (t % 4 == 1)
	{
		result[t] = a[t] - b[t];
	}
	else if (t % 4 == 2)
	{
		result[t] = a[t] * b[t];
	}
	else if (t % 4 == 3)
	{
		result[t] = a[t] % b[t];
	}
}

__global__
void add(int* a, int* b, int* result)
{
	const unsigned int t = (blockIdx.x * blockDim.x) + threadIdx.x;
	result[t] = a[t] + b[t];
}

__global__
void subtract(int* a, int* b, int* result)
{
	const unsigned int t = (blockIdx.x * blockDim.x) + threadIdx.x;
	result[t] = a[t] - b[t];
}

__global__
void mult(int* a, int* b, int* result)
{
	const unsigned int t = (blockIdx.x * blockDim.x) + threadIdx.x;
	result[t] = a[t] * b[t];
}

__global__
void mod(int* a, int* b, int* result)
{
	const unsigned int t = (blockIdx.x * blockDim.x) + threadIdx.x;
	result[t] = a[t] % b[t];
}

void display(const char* descriptor, const int* arr, const int size)
{
	printf("%10s: [", descriptor);
	for (int i = 0; i < size; ++i)
	{
		printf("%2d", arr[i]);
		if (i < size - 1)
		{
			printf(", ");
		}
	}
	printf("]\n");
}

int main(int argc, char** argv)
{
	// Read command line arguments
	int totalThreads = (1 << 20);
	int blockSize = 256;
	if (argc >= 2) 
	{
		totalThreads = atoi(argv[1]);
	}
	if (argc >= 3) 
	{
		blockSize = atoi(argv[2]);
	}

	// Validate command line arguments
	int numBlocks = totalThreads / blockSize;
	if (numBlocks % 4 != 0)
	{
		printf("Warning: need number of blocks to be divisible by 4, exiting\n");
		return 1;
	}
	if (totalThreads % blockSize != 0) 
	{
		++numBlocks;
		totalThreads = numBlocks * blockSize;
		
		printf("Warning: Total thread count is not evenly divisible by the block size\n");
		printf("The total number of threads will be rounded up to %d\n", totalThreads);
	}
	const int numThreads = totalThreads / numBlocks;

	// Allocate and populate memory on cpu
	const int arraySize = sizeof(int) * totalThreads;
	int* cpuA = (int*)malloc(arraySize);
	int* cpuB = (int*)malloc(arraySize);
	for (int i = 0; i < totalThreads; ++i)
	{
		cpuA[i] = i;
		cpuB[i] = rand() % 4;
	}

	// Allocate memory on gpu
	int* a = nullptr;
	int* b = nullptr;
	int* result = nullptr;
	cudaMalloc((void**)&a, arraySize);
	cudaMalloc((void**)&b, arraySize);
	cudaMalloc((void**)&result, arraySize);
	cudaMemcpy(a, cpuA, arraySize, cudaMemcpyHostToDevice);
	cudaMemcpy(b, cpuB, arraySize, cudaMemcpyHostToDevice);

	// Calculate and display kernel branching results
	auto startBranch = std::chrono::high_resolution_clock::now();
	branch<<<numBlocks, numThreads>>>(a, b, result);
	auto stopBranch = std::chrono::high_resolution_clock::now();

	const int blocksPerKernel = numBlocks / 4;

	// Calculate and display add results
	auto start = std::chrono::high_resolution_clock::now();
	add<<<blocksPerKernel, numThreads>>>(a, b, result);

	// Calculate and display subtract results
	subtract<<<blocksPerKernel, numThreads>>> (a + blocksPerKernel * numThreads, b + blocksPerKernel * numThreads, result + blocksPerKernel * numThreads);

	// Calculate and display mult results
	mult<<<blocksPerKernel, numThreads>>> (a + blocksPerKernel * numThreads * 2, b + blocksPerKernel * numThreads * 2, result + blocksPerKernel * numThreads * 2);

	// Calculate and display mod results
	mod<<<blocksPerKernel, numThreads>>> (a + blocksPerKernel * numThreads * 3, b + blocksPerKernel * numThreads * 3, result + blocksPerKernel * numThreads * 3);
	auto stop = std::chrono::high_resolution_clock::now();

	std::cout << std::chrono::duration_cast<std::chrono::nanoseconds>(stopBranch - startBranch).count() << "\n";
	std::cout << std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count() << "\n";

	cudaFree(a);
	cudaFree(b);
	cudaFree(result);
	free(cpuA);
	free(cpuB);
}
