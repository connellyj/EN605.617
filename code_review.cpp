// -GE CONFIDENTIAL-
// Type: Source Code
//
// Copyright (c) 2021, GE Healthcare
// All Rights Reserved
//
// This unpublished material is proprietary to GE Healthcare. The methods and
// techniques described herein are considered trade secrets and/or
// confidential. Reproduction or distribution, in whole or in part, is
// forbidden except by express written permission of GE Healthcare.

// Most of the code for this project is proprietary GE code, so I will only include snippets here
// fft lib: https://github.com/clMathLibraries/clFFT
// some code based on the clFFT examples

// The goal is to replace the CPU based fft with a GPU one to see if overall image reconstruction is improved


// initialize OpenCL context and queue for entire application
// context and queue are reused across different processing threads/sections
void Application::initOpenCL()
{
    cl_int err;
    cl_platform_id platform = 0;
    cl_device_id device = 0;

    /* Setup OpenCL environment. */
    err = clGetPlatformIDs(1, &platform, NULL);
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);

    cl_context_properties props[3] = { CL_CONTEXT_PLATFORM, (cl_context_properties)platform, 0 };
    myContext = std::make_shared<cl_context>(clCreateContext(props, 1, &device, NULL, NULL, &err));
    myQueue = std::make_shared<cl_command_queue>(clCreateCommandQueue(*myContext, device, 0, &err));

    /* Setup clFFT. */
    clfftSetupData fftSetup;
    err = clfftInitSetupData(&fftSetup);
    err = clfftSetup(&fftSetup);
}

// 1D FFT
// MDArray is a multi dimensional array library
void KSpaceTransformer::fft1d(MDArray::ComplexFloatVector& output, const MDArray::ComplexFloatVector& input, const MDArray::ComplexFloatVector& filter)
{
    const int transformLength = output.extent(firstDim);
    int points = input.extent(firstDim);
    ptrdiff_t transformOffset = (transformLength - input.extent(firstDim)) / 2;
    output(MDArray::Range::all()) = 0;
    std::complex<float>* transformData = output.dataZero() + transformOffset;
    const std::complex<float>* inputData = input.dataZero();
    const std::complex<float>* filterData = filter.dataZero();
    for(int i = 0; i < points; ++i)
    {
        *transformData = (*inputData) * (*filterData);

        ++transformData;
        inputData += input.stride(0);
        filterData += filter.stride(0);
    }

    cl_int err;
    cl_mem bufX;
    int ret = 0;
    size_t N = output.extent(firstDim);
    size_t size = N * 2 * sizeof(float);

    // FFT library realted declarations 
    clfftPlanHandle planHandle;
    clfftDim dim = CLFFT_1D;
    size_t clLengths[1] = { N };

    // Prepare OpenCL memory objects and place data inside them.
    bufX = clCreateBuffer(*myContext, CL_MEM_READ_WRITE, size, NULL, &err);

    err = clEnqueueWriteBuffer(*myQueue, bufX, CL_TRUE, 0,
        size, output.dataZero(), 0, NULL, NULL);
    checkErr(err, "clEnqueueWriteBuffer");

    // Create a default plan for a complex FFT. 
    err = clfftCreateDefaultPlan(&planHandle, *myContext, dim, clLengths);
    checkErr(err, "clfftCreateDefaultPlan");

    // Set plan parameters. 
    err = clfftSetPlanPrecision(planHandle, CLFFT_SINGLE);
    checkErr(err, "clfftSetPlanPrecision");
    err = clfftSetLayout(planHandle, CLFFT_COMPLEX_INTERLEAVED, CLFFT_COMPLEX_INTERLEAVED);
    checkErr(err, "clfftSetLayout");
    err = clfftSetResultLocation(planHandle, CLFFT_INPLACE);
    checkErr(err, "clfftSetResultLocation");

    // Bake the plan. 
    err = clfftBakePlan(planHandle, 1, &(*myQueue), NULL, NULL);
    checkErr(err, "clfftBakePlan");

    // Execute the plan.
    cl_event event;
    err = clfftEnqueueTransform(planHandle, CLFFT_FORWARD, 1, &(*myQueue), 0, NULL, &event, &bufX, NULL, NULL);
    checkErr(err, "clfftEnqueueTransform");

    // Fetch results of calculations.
    err = clEnqueueReadBuffer(*myQueue, bufX, CL_TRUE, 0, size, output.dataZero(), 1, &event, NULL);
    checkErr(err, "clEnqueueReadBuffer");

    // Release OpenCL memory objects.
    clReleaseMemObject(bufX);

    // Release the plan.
    err = clfftDestroyPlan(&planHandle);
    checkErr(err, "clfftDestroyPlan");
}

// 2D FFT
// MDArray is a multi dimensional array library
void KSpaceTransformer::fft2d()
{
    fill(outputImageSpace, inputKSpace, kSpaceFilter);

    cl_int err;
    cl_mem bufX;
    int ret = 0;
    size_t N0 = outputImageSpace.extent(firstDim);
    size_t N1 = outputImageSpace.extent(secondDim);
    size_t size = N0 * N1 * 2 * sizeof(float);

    // FFT library realted declarations
    clfftPlanHandle planHandle;
    clfftDim dim = CLFFT_2D;
    size_t clLengths[2] = { N0, N1 };

    // Prepare OpenCL memory objects and place data inside them.
    bufX = clCreateBuffer(*myContext, CL_MEM_READ_WRITE, size, NULL, &err);

    err = clEnqueueWriteBuffer(*myQueue, bufX, CL_TRUE, 0,
        size, outputImageSpace.dataZero(), 0, NULL, NULL);
    checkErr(err, "clEnqueueWriteBuffer");

    // Create a default plan for a complex FFT.
    err = clfftCreateDefaultPlan(&planHandle, *myContext, dim, clLengths);
    checkErr(err, "clfftCreateDefaultPlan");

    // Set plan parameters.
    err = clfftSetPlanPrecision(planHandle, CLFFT_SINGLE);
    checkErr(err, "clfftSetPlanPrecision");
    err = clfftSetLayout(planHandle, CLFFT_COMPLEX_INTERLEAVED, CLFFT_COMPLEX_INTERLEAVED);
    checkErr(err, "clfftSetLayout");
    err = clfftSetResultLocation(planHandle, CLFFT_INPLACE);
    checkErr(err, "clfftSetResultLocation");

    // Bake the plan.
    err = clfftBakePlan(planHandle, 1, &(*myQueue), NULL, NULL);
    checkErr(err, "clfftBakePlan");

    // Execute the plan.
    cl_event event;
    err = clfftEnqueueTransform(planHandle, CLFFT_FORWARD, 1, &(*myQueue), 0, NULL, &event, &bufX, NULL, NULL);
    checkErr(err, "clfftEnqueueTransform");

    // Fetch results of calculations.
    err = clEnqueueReadBuffer(*myQueue, bufX, CL_TRUE, 0, size, outputImageSpace.dataZero(), 1, &event, NULL);
    checkErr(err, "clEnqueueReadBuffer");

    // Release OpenCL memory objects.
    clReleaseMemObject(bufX);

    // Release the plan.
    err = clfftDestroyPlan(&planHandle);
    checkErr(err, "clfftDestroyPlan");
}

// helper method for error catching
void KSpaceTransformer::checkErr(cl_int err, const char * name)
{
    if(err != CL_SUCCESS) {
        std::cerr << "ERROR: " << name << " (" << err << ")" << std::endl;
        exit(EXIT_FAILURE);
    }
}

// helper method to zero fill the data and apply filter before the fft
void KSpaceTransformer::fill(ComplexFloatMatrix& output, const ComplexFloatMatrix& input, const ComplexFloatMatrix& filter)
{
    int points = input.extent(firstDim);
    const int transformLength = output.extent(firstDim);
    ptrdiff_t transformOffset = (transformLength - input.extent(firstDim)) / 2;
    output(MDArray::Range::all(), MDArray::Range::all()) = 0;
    for(int y = 0; y < input.extent(secondDim); ++y)
    {
        std::complex<float>* transformData = output(MDArray::Range::all(), y).dataZero() + transformOffset;
        const std::complex<float>* inputData = input(MDArray::Range::all(), y).dataZero();
        const std::complex<float>* filterData = filter(MDArray::Range::all(), y).dataZero();
        for(int i = 0; i < points; ++i)
        {
            *transformData = (*inputData) * (*filterData);

            ++transformData;
            inputData += input.stride(0);
            filterData += filter.stride(0);
        }
    }
}
