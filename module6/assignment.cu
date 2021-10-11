// Based on the work of Andrew Krepps
// Modifications by Julia Connelly

#include <chrono>
#include <iostream>
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

__global__
void add_reg(int* a, int* b, int* result)
{
	const unsigned int t = (blockIdx.x * blockDim.x) + threadIdx.x;
	int at = a[t];
	int bt = b[t];
	result[t] = at + bt;
}

__global__
void subtract_reg(int* a, int* b, int* result)
{
	const unsigned int t = (blockIdx.x * blockDim.x) + threadIdx.x;
	int at = a[t];
	int bt = b[t];
	result[t] = at - bt;
}

__global__
void mult_reg(int* a, int* b, int* result)
{
	const unsigned int t = (blockIdx.x * blockDim.x) + threadIdx.x;
	int at = a[t];
	int bt = b[t];
	result[t] = at * bt;
}

__global__
void mod_reg(int* a, int* b, int* result)
{
	const unsigned int t = (blockIdx.x * blockDim.x) + threadIdx.x;
	int at = a[t];
	int bt = b[t];
	result[t] = at % bt;
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

void fill_dummy_num(int* a, int* b, const int n)
{
	for (int i = 0; i < n; ++i)
	{
		a[i] = i;
		b[i] = rand() % 4;
	}
}

void baseline_test(const int n, const int num_threads, const int num_blocks, const int display_flag)
{
	const auto start = std::chrono::high_resolution_clock::now();

	// Fill host memory
	const int array_size = sizeof(int) * n;
	int* a = (int*)malloc(array_size);
	int* b = (int*)malloc(array_size);
	int* r = (int*)malloc(array_size);
	fill_dummy_num(a, b, n);

	// Allocate device memory
	int* ag;
	int* bg;
	int* result;
	cudaMalloc((void**)&ag, array_size);
	cudaMalloc((void**)&bg, array_size);
	cudaMalloc((void**)&result, array_size);

	// Copy input to device memory
	cudaMemcpy(ag, a, array_size, cudaMemcpyHostToDevice);
	cudaMemcpy(bg, b, array_size, cudaMemcpyHostToDevice);

	// Execute kernels
	add<<<num_blocks, num_threads>>>(ag, bg, result);
	if(!display_flag)
	{
		subtract<<<num_blocks, num_threads>>>(ag, bg, result);
		mult<<<num_blocks, num_threads>>>(ag, bg, result);
		mod<<<num_blocks, num_threads>>>(ag, bg, result);
	}

	// Display results
	if(display_flag)
	{
		printf("*** baseline ***: \n");
		cudaMemcpy(r, result, array_size, cudaMemcpyDeviceToHost);
		display("a", a, n);
		display("b", b, n);
		display("r", r, n);
	}

	// Clear device memory
	cudaFree(ag);
	cudaFree(bg);
	cudaFree(result);

	const auto end = std::chrono::high_resolution_clock::now();
	if (!display_flag)
	{
		printf("*** baseline ***: \n");
	}
	std::cout << "  time: " << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << " microseconds\n";

	// Free host memory
	free(a);
	free(b);
}

void register_test(const int n, const int num_threads, const int num_blocks, const int display_flag)
{
	const auto start = std::chrono::high_resolution_clock::now();

	// Fill host memory
	const int array_size = sizeof(int) * n;
	int* a = (int*)malloc(array_size);
	int* b = (int*)malloc(array_size);
	int* r = (int*)malloc(array_size);
	fill_dummy_num(a, b, n);

	// Allocate device memory
	int* ag;
	int* bg;
	int* result;
	cudaMalloc((void**)&ag, array_size);
	cudaMalloc((void**)&bg, array_size);
	cudaMalloc((void**)&result, array_size);

	// Copy input to device memory
	cudaMemcpy(ag, a, array_size, cudaMemcpyHostToDevice);
	cudaMemcpy(bg, b, array_size, cudaMemcpyHostToDevice);

	// Execute kernels
	add_reg<<<num_blocks, num_threads>>>(ag, bg, result);
	if(!display_flag)
	{
		subtract_reg<<<num_blocks, num_threads>>>(ag, bg, result);
		mult_reg<<<num_blocks, num_threads>>>(ag, bg, result);
		mod_reg<<<num_blocks, num_threads>>>(ag, bg, result);
	}

	// Display results
	if(display_flag)
	{
		printf("*** register ***: \n");
		cudaMemcpy(r, result, array_size, cudaMemcpyDeviceToHost);
		display("a", a, n);
		display("b", b, n);
		display("r", r, n);
	}

	// Clear device memory
	cudaFree(ag);
	cudaFree(bg);
	cudaFree(result);

	const auto end = std::chrono::high_resolution_clock::now();
	if (!display_flag)
	{
		printf("*** register ***: \n");
	}
	std::cout << "  time: " << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << " microseconds\n";

	// Free host memory
	free(a);
	free(b);
}

int main(int argc, char** argv)
{
	// Read command line arguments
	int total_threads = (1 << 20);
	int block_size = 1024;
	int display_flag = 0;
	if (argc >= 2)
	{
		total_threads = atoi(argv[1]);
	}
	if (argc >= 3)
	{
		block_size = atoi(argv[2]);
	}
	if (argc >= 4)
	{
		display_flag = atoi(argv[3]);
	}

	// Calculate number of threads and blocks to use for tests
	int num_blocks = total_threads / block_size;
	if (total_threads % block_size != 0)
	{
		num_blocks++;
		total_threads = num_blocks * block_size;
	}
	const int num_threads = total_threads / num_blocks;
	printf("total threads: %d\nthreads per block: %d\nblocks: %d\n\n", total_threads, num_threads, num_blocks);

	// Warmup device
	printf("\nstart warmup\n");
	baseline_test(total_threads, num_threads, num_blocks, 0);
	printf("end warmup\n\n");

	// Run tests
	baseline_test(total_threads, num_threads, num_blocks, display_flag);
	register_test(total_threads, num_threads, num_blocks, display_flag);
}
