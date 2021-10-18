// Based on the work of Andrew Krepps
// Modifications by Julia Connelly

#include <chrono>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>

#include <cuda.h> 

__global__
void add(int* a, int* b, int* result)
{
	const unsigned int t = (blockIdx.x * blockDim.x) + threadIdx.x;
	int at = a[t];
	int bt = b[t];
	result[t] = at + bt;
}

__global__
void subtract(int* a, int* b, int* result)
{
	const unsigned int t = (blockIdx.x * blockDim.x) + threadIdx.x;
	int at = a[t];
	int bt = b[t];
	result[t] = at - bt;
}

__global__
void mult(int* a, int* b, int* result)
{
	const unsigned int t = (blockIdx.x * blockDim.x) + threadIdx.x;
	int at = a[t];
	int bt = b[t];
	result[t] = at * bt;
}

__global__
void mod(int* a, int* b, int* result)
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

void display(const char* description, const int* a, const int* b, const int* r, const int n, const float time, const int display_flag)
{
	printf("*** %s ***: \n", description);
	if (display_flag)
	{
		display("a", a, n);
		display("b", b, n);
		display("r", r, n);
	}
	printf("      time: %f\n", time);
}

void fill_dummy_num(int* a, int* b, const int n)
{
	for (int i = 0; i < n; ++i)
	{
		a[i] = i;
		b[i] = rand() % 4;
	}
}

void execute(int* a_out, int* b_out, int* r_out, const int num_blocks, const int num_threads, const int display_flag)
{
	add<<<num_blocks, num_threads>>>(a_out, b_out, r_out);
	if(!display_flag)
	{
		subtract<<<num_blocks, num_threads>>>(a_out, b_out, r_out);
		mult<<<num_blocks, num_threads>>>(a_out, b_out, r_out);
		mod<<<num_blocks, num_threads>>>(a_out, b_out, r_out);
	}
}

void execute(int* a_out, int* b_out, int* r_out, cudaStream_t stream, const int num_blocks, const int num_threads, const int display_flag)
{
	add<<<num_blocks, num_threads, 1, stream>>>(a_out, b_out, r_out);
	if(!display_flag)
	{
		subtract<<<num_blocks, num_threads, 1, stream>>>(a_out, b_out, r_out);
		mult<<<num_blocks, num_threads, 1, stream>>>(a_out, b_out, r_out);
		mod<<<num_blocks, num_threads, 1, stream>>>(a_out, b_out, r_out);
	}
}

void baseline_test(const int n, const int num_threads, const int num_blocks, const int display_flag)
{
	// Fill host memory
	const int array_size = sizeof(int) * n;
	int* a, * b, * r;
	cudaHostAlloc((void**)&a, array_size, cudaHostAllocDefault);
	cudaHostAlloc((void**)&b, array_size, cudaHostAllocDefault);
	cudaHostAlloc((void**)&r, array_size, cudaHostAllocDefault);
	fill_dummy_num(a, b, n);

	// Setup timing events
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);

	// Copy to device
	int *a_out, *b_out, *r_out;
	cudaMalloc((void**)&a_out, array_size);
	cudaMalloc((void**)&b_out, array_size);
	cudaMalloc((void**)&r_out, array_size);
	cudaMemcpy(a_out, a, array_size, cudaMemcpyHostToDevice);
	cudaMemcpy(b_out, b, array_size, cudaMemcpyHostToDevice);
	
	// Execute kernels
	execute(a_out, b_out, r_out, num_blocks, num_threads, display_flag);

	// Copy results out
	cudaMemcpy(r, r_out, array_size, cudaMemcpyDeviceToHost);

	// Synchronize
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	float elapsed_time;
	cudaEventElapsedTime(&elapsed_time, start, stop);

	// Display results
	display("baseline", a, b, r, n, elapsed_time, display_flag);

	// Clear device memory
	cudaFree(a_out);
	cudaFree(b_out);
	cudaFree(r_out);

	// Free host memory
	cudaFreeHost(a);
	cudaFreeHost(b);
	cudaFreeHost(r);
}

void stream_test(const int n, const int num_threads, const int num_blocks, const int display_flag)
{
	// Fill host memory
	const int array_size = sizeof(int) * n;
	int *a, *b, *r;
	cudaHostAlloc((void**)&a, array_size, cudaHostAllocDefault);
	cudaHostAlloc((void**)&b, array_size, cudaHostAllocDefault);
	cudaHostAlloc((void**)&r, array_size, cudaHostAllocDefault);
	fill_dummy_num(a, b, n);

	// Setup timing events
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);

	// Allocate device memory
	int *a_out, *b_out, *r_out;
	cudaMalloc((void**)&a_out, array_size);
	cudaMalloc((void**)&b_out, array_size);
	cudaMalloc((void**)&r_out, array_size);

	// Copy async to device
	cudaStream_t stream;
	cudaStreamCreate(&stream);
	cudaMemcpyAsync(a_out, a, array_size, cudaMemcpyHostToDevice, stream);
	cudaMemcpyAsync(b_out, b, array_size, cudaMemcpyHostToDevice, stream);

	// Execute kernels
	execute(a_out, b_out, r_out, stream, num_blocks, num_threads, display_flag);

	// Copy results out
	cudaMemcpyAsync(r, r_out, array_size, cudaMemcpyDeviceToHost, stream);

	// Synchronize
	cudaStreamSynchronize(stream);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	float elapsed_time;
	cudaEventElapsedTime(&elapsed_time, start, stop);

	// Display results
	display("stream", a, b, r, n, elapsed_time, display_flag);

	// Clear device memory
	cudaFree(a_out);
	cudaFree(b_out);
	cudaFree(r_out);

	// Free host memory
	cudaFreeHost(a);
	cudaFreeHost(b);
	cudaFreeHost(r);
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
	stream_test(total_threads, num_threads, num_blocks, display_flag);
}
