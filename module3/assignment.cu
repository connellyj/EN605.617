// Based on the work of Andrew Krepps
// Modifications by Julia Connelly

#include <stdio.h>
#include <stdlib.h>

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
	if (totalThreads % blockSize != 0) 
	{
		++numBlocks;
		totalThreads = numBlocks * blockSize;
		
		printf("Warning: Total thread count is not evenly divisible by the block size\n");
		printf("The total number of threads will be rounded up to %d\n", totalThreads);
	}
	const int numThreads = totalThreads / numBlocks;
	printf("\ntotal threads: %d\nthreads per block: %d\nblocks: %d\n\n", totalThreads, numThreads, numBlocks);

	// Allocate and populate memory on cpu
	const int arraySize = sizeof(int) * totalThreads;
	int* cpuA = (int*)malloc(arraySize);
	int* cpuB = (int*)malloc(arraySize);
	int* cpuResult = (int*)malloc(arraySize);
	for (int i = 0; i < totalThreads; ++i)
	{
		cpuA[i] = i;
		cpuB[i] = rand() % 4;
	}

	// Allocate memory on gpu
	int* a;
	int* b;
	int* result;
	cudaMalloc((void**)&a, arraySize);
	cudaMalloc((void**)&b, arraySize);
	cudaMalloc((void**)&result, arraySize);
	cudaMemcpy(a, cpuA, arraySize, cudaMemcpyHostToDevice);
	cudaMemcpy(b, cpuB, arraySize, cudaMemcpyHostToDevice);

	// Calculate and display add results
	add<<<numBlocks, numThreads>>>(a, b, result);
	cudaMemcpy(cpuResult, result, arraySize, cudaMemcpyDeviceToHost);
	display("a", cpuA, totalThreads);
	display("b", cpuB, totalThreads);
	display("add", cpuResult, totalThreads);
	printf("\n");

	// Calculate and display subtract results
	subtract<<<numBlocks, numThreads>>> (a, b, result);
	cudaMemcpy(cpuResult, result, arraySize, cudaMemcpyDeviceToHost);
	display("a", cpuA, totalThreads);
	display("b", cpuB, totalThreads);
	display("subtract", cpuResult, totalThreads);
	printf("\n");

	// Calculate and display mult results
	mult<<<numBlocks, numThreads>>> (a, b, result);
	cudaMemcpy(cpuResult, result, arraySize, cudaMemcpyDeviceToHost);
	display("a", cpuA, totalThreads);
	display("b", cpuB, totalThreads);
	display("mult", cpuResult, totalThreads);
	printf("\n");

	// Calculate and display mod results
	mod<<<numBlocks, numThreads>>> (a, b, result);
	cudaMemcpy(cpuResult, result, arraySize, cudaMemcpyDeviceToHost);
	display("a", cpuA, totalThreads);
	display("b", cpuB, totalThreads);
	display("mod", cpuResult, totalThreads);
	printf("\n");

	cudaFree(a);
	cudaFree(b);
	cudaFree(result);
	free(cpuA);
	free(cpuB);
	free(cpuResult);
}
