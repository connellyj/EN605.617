// Based on the work of Andrew Krepps
// Modifications by Julia Connelly

#include <chrono>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>

__global__
void caesar(char* message, char* cipher, const int n, const int offset)
{
	const unsigned int t = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (t < n)
	{
		cipher[t] = message[t] + offset;
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

void fill_dummy_str(char* dest, const int n)
{
	for (int i = 0; i < n - 1; ++i)
	{
		dest[i] = 'a' + (i % 26);
	}
	dest[n - 1] = '\0';
}

void fill_dummy_num(int* a, int* b, const int n)
{
	for (int i = 0; i < n; ++i)
	{
		a[i] = i;
		b[i] = rand() % 4;
	}
}

void cipher_on_gpu(char* host_message, char* host_cipher, char* host_undo, const int n, const int offset, const int num_threads, const int num_blocks)
{
	// Allocate device memory
	const int array_size = sizeof(char) * n;
	char* message;
	char* cipher;
	char* undo;
	cudaMalloc((void**)&message, array_size);
	cudaMalloc((void**)&cipher, array_size);
	cudaMalloc((void**)&undo, array_size);

	// Copy input to device memory
	cudaMemcpy(message, host_message, n, cudaMemcpyHostToDevice);

	// Execute kernels
	// -1 to avoid the null terminating char
	caesar<<<num_blocks, num_threads>>>(message, cipher, n - 1, offset);
	caesar<<<num_blocks, num_threads>>>(cipher, undo, n - 1, offset * -1);

	// Copy results out
	cudaMemcpy(host_cipher, cipher, n, cudaMemcpyDeviceToHost);
	cudaMemcpy(host_undo, undo, n, cudaMemcpyDeviceToHost);

	// Clear device memory
	cudaFree(message);
	cudaFree(cipher);
	cudaFree(undo);
}

void math_on_gpu(const int* host_a, const int* host_b, const int n, const int num_threads, const int num_blocks)
{
	// Allocate device memory
	const int array_size = sizeof(int) * n;
	int* a;
	int* b;
	int* result;
	cudaMalloc((void**)&a, array_size);
	cudaMalloc((void**)&b, array_size);
	cudaMalloc((void**)&result, array_size);

	// Copy input to device memory
	cudaMemcpy(a, host_a, n, cudaMemcpyHostToDevice);
	cudaMemcpy(b, host_b, n, cudaMemcpyHostToDevice);

	// Execute kernels
	add<<<num_blocks, num_threads>>>(a, b, result);
	subtract<<<num_blocks, num_threads>>>(a, b, result);
	mult<<<num_blocks, num_threads>>>(a, b, result);
	mod<<<num_blocks, num_threads>>>(a, b, result);

	// Clear device memory
	cudaFree(a);
	cudaFree(b);
	cudaFree(result);
}

void cipher_host_test(const int n, const int offset, const int num_threads, const int num_blocks, const int display_flag)
{
	const auto start = std::chrono::high_resolution_clock::now();

	// Fill host memory
	const int array_size = sizeof(char) * n;
	char* message = (char*)malloc(array_size);
	fill_dummy_str(message, n);
	char* cipher = (char*)malloc(array_size);
	char* undo = (char*)malloc(array_size);

	// Execute kernels
	cipher_on_gpu(message, cipher, undo, n, offset, num_threads, num_blocks);

	const auto end = std::chrono::high_resolution_clock::now();

	printf("*** host ***: \n");
	if (display_flag)
	{
		printf(" input: %s\n", message);
		printf("  undo: %s\n", undo);
		printf("output: %s\n", cipher);
	}
	std::cout << "  time: " << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << " microseconds\n";
}

void math_host_test(const int n, const int num_threads, const int num_blocks)
{
	const auto start = std::chrono::high_resolution_clock::now();

	// Fill host memory
	const int array_size = sizeof(int) * n;
	int* a = (int*)malloc(array_size);
	int* b = (int*)malloc(array_size);
	fill_dummy_num(a, b, n);

	// Execute kernels
	math_on_gpu(a, b, n, num_threads, num_blocks);

	const auto end = std::chrono::high_resolution_clock::now();

	printf("*** host ***: \n");
	std::cout << "  time: " << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << " microseconds\n";

	// Free host memory
	free(a);
	free(b);
}

void cipher_pinned_test(const int n, const int offset, const int num_threads, const int num_blocks, const int display_flag)
{
	const auto start = std::chrono::high_resolution_clock::now();

	// Fill pinned host memory
	const int array_size = sizeof(char) * n;
	char* pinned_message;
	char* pinned_cipher;
	char* pinned_undo;
	cudaMallocHost((void**)&pinned_message, array_size);
	fill_dummy_str(pinned_message, n);
	cudaMallocHost((void**)&pinned_cipher, array_size);
	cudaMallocHost((void**)&pinned_undo, array_size);

	// Execute kernels
	cipher_on_gpu(pinned_message, pinned_cipher, pinned_undo, n, offset, num_threads, num_blocks);

	const auto end = std::chrono::high_resolution_clock::now();

	printf("*** pinned ***: \n");
	if (display_flag)
	{
		printf(" input: %s\n", pinned_message);
		printf("  undo: %s\n", pinned_undo);
		printf("output: %s\n", pinned_cipher);
	}
	std::cout << "  time: " << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << " microseconds\n";

	// Free host memory
	cudaFreeHost(pinned_message);
	cudaFreeHost(pinned_cipher);
	cudaFreeHost(pinned_undo);
}

void math_pinned_test(const int n, const int num_threads, const int num_blocks)
{
	const auto start = std::chrono::high_resolution_clock::now();

	// Fill host memory
	const int array_size = sizeof(int) * n;
	int* a;
	int* b;
	cudaMallocHost((void**)&a, array_size);
	cudaMallocHost((void**)&b, array_size);
	fill_dummy_num(a, b, n);

	// Execute kernels
	math_on_gpu(a, b, n, num_threads, num_blocks);

	const auto end = std::chrono::high_resolution_clock::now();

	printf("*** pinned ***: \n");
	std::cout << "  time: " << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << " microseconds\n";

	// Free host memory
	cudaFreeHost(a);
	cudaFreeHost(b);
}

void cipher_cpu_test(const int n, const int offset, const int display_flag)
{
	const auto start = std::chrono::high_resolution_clock::now();

	// Fill host memory
	const int array_size = sizeof(char) * n;
	char* message = (char*)malloc(array_size);
	fill_dummy_str(message, n);
	char* cipher = (char*)malloc(array_size);
	char* undo = (char*)malloc(array_size);

	// Compute cipher
	// -1 to avoid the null terminating char
	for(int i = 0; i < n - 1; ++i)
	{
		cipher[i] = message[i] + offset;
	}

	// Compute inverse cipher
	// -1 to avoid the null terminating char
	for (int i = 0; i < n - 1; ++i)
	{
		undo[i] = cipher[i] - offset;
	}

	const auto end = std::chrono::high_resolution_clock::now();

	printf("*** cpu ***: \n");
	if (display_flag)
	{
		printf(" input: %s\n", message);
		printf("  undo: %s\n", undo);
		printf("output: %s\n", cipher);
	}
	std::cout << "  time: " << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << " microseconds\n";

	// Free host memory
	free(message);
	free(cipher);
	free(undo);
}

int main(int argc, char** argv)
{
	const int offset = 3;
	const int block_size = 1024;

	// Read command line arguments
	int total_threads = (1 << 20);
	int display_flag = 0;
	if (argc >= 2)
	{
		total_threads = atoi(argv[1]);
	}
	if (argc >= 3)
	{
		display_flag = atoi(argv[2]);
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
	// Without warmpup, pinned memory times are terrible
	printf("\nWARMUP:\n");
	math_pinned_test(total_threads, num_threads, num_blocks);

	// Run math tests
	printf("\nMATH:\n");
	math_host_test(total_threads, num_threads, num_blocks);
	math_pinned_test(total_threads, num_threads, num_blocks);

	// Run cipher tests
	printf("\nCIPHER:\n");
	cipher_cpu_test(total_threads, offset, display_flag);
	cipher_host_test(total_threads, offset, num_threads, num_blocks, display_flag);
	cipher_pinned_test(total_threads, offset, num_threads, num_blocks, display_flag);
}
