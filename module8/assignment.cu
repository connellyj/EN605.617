// Found at http://techqa.info/programming/question/36889333/cuda-cufft-2d-example
// Modified by Julia Connelly

// compile on windows:
// nvcc assignment.cu -o assignment "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.4\lib\x64\cufft.lib" "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.4\lib\x64\cublas.lib"

#include <iostream>
#include <stdio.h>

#include "cublas.h"
#include "cuda.h"
#include "cufft.h"

#define index(i,j,ld) (((j)*(ld))+(i))

typedef cufftComplex Complex;

void display_matrix(const Complex* mat, const int n)
{
    for (int i = 0; i < n * n; i = i + n)
    {
        for (int j = 0; j < n; j++) {
            std::cout << mat[i + j].x << " ";
        }
        std::cout << std::endl;
    }
    std::cout << "----------------" << std::endl << std::endl;
}

void display_matrix(const float* mat, const int n)
{
    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            std::cout << mat[index(i, j, n)] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << "----------------" << std::endl << std::endl;
}

void cublas_test(const int n, const int display_flag)
{
    cublasInit();

    // initialize host matrices
    const int mem_size = n * n * sizeof(float);
    float* A = (float*)malloc(mem_size);
    float* B = (float*)malloc(mem_size);
    float* C = (float*)malloc(mem_size);
    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            A[index(i, j, n)] = (float)index(i, j, n);
            B[index(i, j, n)] = (float)index(i, j, n);
        }
    }

    // setup timing events
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // allocate on device
    float* AA; float* BB; float* CC;
    cudaMalloc((void**)&AA, mem_size);
    cudaMalloc((void**)&BB, mem_size);
    cudaMalloc((void**)&CC, mem_size);

    // set matrices and execute mult
    cublasSetMatrix(n, n, sizeof(float), A, n, AA, n);
    cublasSetMatrix(n, n, sizeof(float), B, n, BB, n);
    cublasSgemm('n', 'n', n, n, n, 1, AA, n, BB, n, 0, CC, n);
    cublasGetMatrix(n, n, sizeof(float), CC, n, C, n);

    if (display_flag)
    {
        printf("input matrix 1:\n\n");
        display_matrix(A, n);
        printf("input matrix 2:\n\n");
        display_matrix(B, n);
        printf("multiplied matrix:\n\n");
        display_matrix(C, n);
    }

    // calculate time
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float elapsed_time;
    cudaEventElapsedTime(&elapsed_time, start, stop);
    printf("elapsed time: %f\n", elapsed_time);

    // destroy memory
    free(A);  
    free(B);  
    free(C);
    cudaFree(AA);
    cudaFree(BB);
    cudaFree(CC);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cublasShutdown();
}

void cufft_test(const int n, const int display_flag)
{
    // initialize host matrices
    const int mem_size = sizeof(Complex) * n * n;
    Complex* input_mat = (Complex*)malloc(mem_size);
    Complex* output_mat = (Complex*)malloc(mem_size);
    for (int i = 0; i < n * n; i++)
    {
        input_mat[i].x = 1;
        input_mat[i].y = 0;
    }
    if (display_flag)
    {
        printf("input matrix:\n\n");
        display_matrix(input_mat, n);
    }

    // setup timing events
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // allocate and move data to device
    cufftComplex* d_signal;
    cudaMalloc((void**)&d_signal, mem_size);
    cudaMemcpy(d_signal, input_mat, mem_size, cudaMemcpyHostToDevice);

    // set up fft plan
    cufftHandle plan;
    cufftPlan2d(&plan, n, n, CUFFT_C2C);

    // transform signal
    cufftExecC2C(plan, d_signal, d_signal, CUFFT_FORWARD);
    cudaMemcpy(output_mat, d_signal, mem_size, cudaMemcpyDeviceToHost);
    if (display_flag)
    {
        printf("transformed matrix:\n\n");
        display_matrix(output_mat, n);
    }

    // transform signal back
    cufftExecC2C(plan, d_signal, d_signal, CUFFT_INVERSE);
    cudaMemcpy(output_mat, d_signal, mem_size, cudaMemcpyDeviceToHost);
    if (display_flag)
    {
        printf("un-transformed matrix:\n\n");
        display_matrix(output_mat, n);
    }

    // calculate time
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float elapsed_time;
    cudaEventElapsedTime(&elapsed_time, start, stop);
    printf("elapsed time: %f\n", elapsed_time);

    // destroy memory
    cufftDestroy(plan);
    cudaFree(d_signal);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

int main(int argc, char** argv)
{
    // Read command line arguments
    int n = 5;
    int display_flag = 0;
    if (argc >= 2)
    {
        n = atoi(argv[1]);
    }
    if (argc >= 3)
    {
        display_flag = atoi(argv[2]);
    }
    else if (n < 6)
    {
        display_flag = 1;
    }

    printf("\n\nCUBLAS\n----------------\n\n");
    cublas_test(n, display_flag);
    printf("\n\nCUFFT\n----------------\n\n");
    cufft_test(n, display_flag);
}
