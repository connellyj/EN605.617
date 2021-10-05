// Based on the work of Andrew Krepps
// Modifications by Julia Connelly

#include <chrono>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>

#define DISPLAY_FLAG 0

#define MAX_SHARED_NUM_ELEMENTS 12288
#define MAX_CONST_NUM_ELEMENTS 16384
#define NUM_ELEMENTS 12288
#define BLOCK_SIZE 512

__constant__ short A_CONST[NUM_ELEMENTS];
__constant__ short B_CONST[NUM_ELEMENTS];

void display(const char* descriptor, const short* arr)
{
	printf("%10s: [", descriptor);
	for (long i = 0; i < NUM_ELEMENTS; ++i)
	{
		printf("%2d", arr[i]);
		if (i < NUM_ELEMENTS - 1)
		{
			printf(", ");
		}
	}
	printf("]\n");
}

void fill_dummy_num(short* a, short* b)
{
	for (long i = 0; i < NUM_ELEMENTS; ++i)
	{
		a[i] = i;  // This could overflow, but I'm ok with that
		b[i] = rand() % 4;
	}
}

__global__
void add(short* a, short* b, short* result)
{
	const unsigned int t = (blockIdx.x * blockDim.x) + threadIdx.x;
	result[t] = a[t] + b[t];
}

__global__
void subtract(short* a, short* b, short* result)
{
	const unsigned int t = (blockIdx.x * blockDim.x) + threadIdx.x;
	result[t] = a[t] - b[t];
}

__global__
void mult(short* a, short* b, short* result)
{
	const unsigned int t = (blockIdx.x * blockDim.x) + threadIdx.x;
	result[t] = a[t] * b[t];
}

__global__
void mod(short* a, short* b, short* result)
{
	const unsigned int t = (blockIdx.x * blockDim.x) + threadIdx.x;
	result[t] = a[t] % b[t];
}

__global__
void add_const(short* result)
{
	const unsigned int t = (blockIdx.x * blockDim.x) + threadIdx.x;
	result[t] = A_CONST[t] + B_CONST[t];
}

__global__
void subtract_const(short* result)
{
	const unsigned int t = (blockIdx.x * blockDim.x) + threadIdx.x;
	result[t] = A_CONST[t] - B_CONST[t];
}

__global__
void mult_const(short* result)
{
	const unsigned int t = (blockIdx.x * blockDim.x) + threadIdx.x;
	result[t] = A_CONST[t] * B_CONST[t];
}

__global__
void mod_const(short* result)
{
	const unsigned int t = (blockIdx.x * blockDim.x) + threadIdx.x;
	result[t] = A_CONST[t] % B_CONST[t];
}

__global__
void add_shared_copy(short* a, short* b, short* result)
{
	__shared__ short a_s[NUM_ELEMENTS];
	__shared__ short b_s[NUM_ELEMENTS];
	const unsigned int t = (blockIdx.x * blockDim.x) + threadIdx.x;

	// copy into shared memory
	a_s[t] = a[t];
	b_s[t] = b[t];
	__syncthreads();

	result[t] = a_s[t] + b_s[t];
}

__global__
void subtract_shared_copy(short* a, short* b, short* result)
{
	__shared__ short a_s[NUM_ELEMENTS];
	__shared__ short b_s[NUM_ELEMENTS];
	const unsigned int t = (blockIdx.x * blockDim.x) + threadIdx.x;

	// copy into shared memory
	a_s[t] = a[t];
	b_s[t] = b[t];
	__syncthreads();

	result[t] = a_s[t] - b_s[t];
}

__global__
void mult_shared_copy(short* a, short* b, short* result)
{
	__shared__ short a_s[NUM_ELEMENTS];
	__shared__ short b_s[NUM_ELEMENTS];
	const unsigned int t = (blockIdx.x * blockDim.x) + threadIdx.x;

	// copy into shared memory
	a_s[t] = a[t];
	b_s[t] = b[t];
	__syncthreads();

	result[t] = a_s[t] * b_s[t];
}

__global__
void mod_shared_copy(short* a, short* b, short* result)
{
	__shared__ short a_s[NUM_ELEMENTS];
	__shared__ short b_s[NUM_ELEMENTS];
	const unsigned int t = (blockIdx.x * blockDim.x) + threadIdx.x;

	// copy into shared memory
	a_s[t] = a[t];
	b_s[t] = b[t];
	__syncthreads();

	result[t] = a_s[t] % b_s[t];
}

void shared_copy_test()
{
	const auto start = std::chrono::high_resolution_clock::now();

	// Fill host memory
	const int array_size = sizeof(short) * NUM_ELEMENTS;
	short* a = (short*)malloc(array_size);
	short* b = (short*)malloc(array_size);
	short* r = (short*)malloc(array_size);
	fill_dummy_num(a, b);

	// Allocate device memory
	short* a_d;
	short* b_d;
	short* result;
	cudaMalloc((void**)&a_d, array_size);
	cudaMalloc((void**)&b_d, array_size);
	cudaMalloc((void**)&result, array_size);

	// Copy input to device memory
	cudaMemcpy(a_d, a, array_size, cudaMemcpyHostToDevice);
	cudaMemcpy(b_d, b, array_size, cudaMemcpyHostToDevice);

	// Execute kernels
	const int num_blocks = NUM_ELEMENTS / BLOCK_SIZE;
	const int num_threads = NUM_ELEMENTS / num_blocks;
	add_shared_copy << <num_blocks, num_threads >> > (a_d, b_d, result);
	if (!DISPLAY_FLAG)
	{
		subtract_shared_copy << <num_blocks, num_threads >> > (a_d, b_d, result);
		mult_shared_copy << <num_blocks, num_threads >> > (a_d, b_d, result);
		mod_shared_copy << <num_blocks, num_threads >> > (a_d, b_d, result);
	}

	if (DISPLAY_FLAG)
	{
		cudaMemcpy(r, result, array_size, cudaMemcpyDeviceToHost);
		display("a", a);
		display("b", b);
		display("r", r);
	}

	// Clear device memory
	cudaFree(a_d);
	cudaFree(b_d);
	cudaFree(result);

	const auto end = std::chrono::high_resolution_clock::now();

	std::cout << "shared copy: " << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << " microseconds\n";

	// Free host memory
	free(a);
	free(b);
	free(r);
}

void constant_copy_test()
{
	const auto start = std::chrono::high_resolution_clock::now();

	// Fill host memory
	const int array_size = sizeof(short) * NUM_ELEMENTS;
	short* a = (short*)malloc(array_size);
	short* b = (short*)malloc(array_size);
	short* r = (short*)malloc(array_size);
	fill_dummy_num(a, b);

	// Allocate device memory
	short* result;
	cudaMalloc((void**)&result, array_size);

	// Copy input to constant device memory
	cudaMemcpyToSymbol(A_CONST, a, array_size);
	cudaMemcpyToSymbol(B_CONST, b, array_size);

	// Execute kernels
	const int num_blocks = NUM_ELEMENTS / BLOCK_SIZE;
	const int num_threads = NUM_ELEMENTS / num_blocks;
	add_const<<<num_blocks, num_threads>>>(result);
	if (!DISPLAY_FLAG)
	{
		subtract_const << <num_blocks, num_threads >> > (result);
		mult_const << <num_blocks, num_threads >> > (result);
		mod_const << <num_blocks, num_threads >> > (result);
	}

	if (DISPLAY_FLAG)
	{
		cudaMemcpy(r, result, array_size, cudaMemcpyDeviceToHost);
		display("a", a);
		display("b", b);
		display("r", r);
	}

	// Clear device memory
	cudaFree(result);

	const auto end = std::chrono::high_resolution_clock::now();

	std::cout << "   constant: " << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << " microseconds\n";

	// Free host memory
	free(a);
	free(b);
	free(r);
}

void baseline_test()
{
	const auto start = std::chrono::high_resolution_clock::now();

	// Fill host memory
	const int array_size = sizeof(short) * NUM_ELEMENTS;
	short* a = (short*)malloc(array_size);
	short* b = (short*)malloc(array_size);
	short* r = (short*)malloc(array_size);
	fill_dummy_num(a, b);

	// Allocate device memory
	short* a_d;
	short* b_d;
	short* result;
	cudaMalloc((void**)&a_d, array_size);
	cudaMalloc((void**)&b_d, array_size);
	cudaMalloc((void**)&result, array_size);

	// Copy input to device memory
	cudaMemcpy(a_d, a, array_size, cudaMemcpyHostToDevice);
	cudaMemcpy(b_d, b, array_size, cudaMemcpyHostToDevice);

	// Execute kernels
	const int num_blocks = NUM_ELEMENTS / BLOCK_SIZE;
	const int num_threads = NUM_ELEMENTS / num_blocks;
	add<<<num_blocks, num_threads>>>(a_d, b_d, result);
	if (!DISPLAY_FLAG)
	{
		subtract << <num_blocks, num_threads >> > (a_d, b_d, result);
		mult << <num_blocks, num_threads >> > (a_d, b_d, result);
		mod << <num_blocks, num_threads >> > (a_d, b_d, result);
	}

	if (DISPLAY_FLAG)
	{
		cudaMemcpy(r, result, array_size, cudaMemcpyDeviceToHost);
		display("a", a);
		display("b", b);
		display("r", r);
	}

	// Clear device memory
	cudaFree(a_d);
	cudaFree(b_d);
	cudaFree(result);

	const auto end = std::chrono::high_resolution_clock::now();

	std::cout << "   baseline: " << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << " microseconds\n";

	// Free host memory
	free(a);
	free(b);
	free(r);
}

int main(int argc, char** argv)
{
	const int num_blocks = NUM_ELEMENTS / BLOCK_SIZE;
	const int num_threads = NUM_ELEMENTS / num_blocks;
	printf("\ntotal threads: %d\nthreads per block: %d\nblocks: %d\n\n", NUM_ELEMENTS, num_threads, num_blocks);

	printf("\nstart warmup\n");
	baseline_test();
	printf("end warmup\n\n");

	baseline_test();
	constant_copy_test();
	shared_copy_test();
}
