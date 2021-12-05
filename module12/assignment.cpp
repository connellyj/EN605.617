//
// Book:      OpenCL(R) Programming Guide
// Authors:   Aaftab Munshi, Benedict Gaster, Timothy Mattson, James Fung, Dan Ginsburg
// ISBN-10:   0-321-74964-2
// ISBN-13:   978-0-321-74964-2
// Publisher: Addison-Wesley Professional
// URLs:      http://safari.informit.com/9780132488006/
//            http://www.openclprogrammingguide.com
//

// raytracer.cpp
//
//    This is a (very) simple raytracer that is intended to demonstrate 
//    using OpenCL buffers.

// modified by Julia connelly
// compile on windows: g++ assignment.cpp -o assignment -I "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.4\include" "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.4\lib\x64\OpenCL.lib"

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "info.hpp"

#define NUM_BUFFER_ELEMENTS 16
#define SUB_BUFFER_SIZE 4

// Function to check and handle OpenCL errors
inline void 
checkErr(cl_int err, const char * name)
{
    if (err != CL_SUCCESS) {
        std::cerr << "ERROR: " <<  name << " (" << err << ")" << std::endl;
        exit(EXIT_FAILURE);
    }
}

///
//	main() for simple buffer and sub-buffer example
//
int main(int argc, char** argv)
{
    cl_int errNum;
    cl_uint numPlatforms;
    cl_uint numDevices;
    cl_platform_id * platformIDs;
    cl_device_id * deviceIDs;
    cl_context context;
    cl_program program;
    std::vector<cl_kernel> kernels;
    std::vector<cl_command_queue> queues;
    std::vector<cl_mem> buffers;
    int * inputOutput;

    std::cout << "Simple buffer and sub-buffer Example" << std::endl;

    // First, select an OpenCL platform to run on.  
    errNum = clGetPlatformIDs(0, NULL, &numPlatforms);
    checkErr( 
        (errNum != CL_SUCCESS) ? errNum : (numPlatforms <= 0 ? -1 : CL_SUCCESS), 
        "clGetPlatformIDs"); 
 
    platformIDs = (cl_platform_id *)alloca(
            sizeof(cl_platform_id) * numPlatforms);

    errNum = clGetPlatformIDs(numPlatforms, platformIDs, NULL);
    checkErr( 
       (errNum != CL_SUCCESS) ? errNum : (numPlatforms <= 0 ? -1 : CL_SUCCESS), 
       "clGetPlatformIDs");

    std::ifstream srcFile("assignment.cl");
    checkErr(srcFile.is_open() ? CL_SUCCESS : -1, "reading assignment.cl");

    std::string srcProg(
        std::istreambuf_iterator<char>(srcFile),
        (std::istreambuf_iterator<char>()));

    const char * src = srcProg.c_str();
    size_t length = srcProg.length();

    deviceIDs = NULL;
    DisplayPlatformInfo(
        platformIDs[0], 
        CL_PLATFORM_VENDOR, 
        "CL_PLATFORM_VENDOR");

    errNum = clGetDeviceIDs(
        platformIDs[0], 
        CL_DEVICE_TYPE_ALL, 
        0,
        NULL,
        &numDevices);
    if (errNum != CL_SUCCESS && errNum != CL_DEVICE_NOT_FOUND)
    {
        checkErr(errNum, "clGetDeviceIDs");
    }       

    deviceIDs = (cl_device_id *)alloca(sizeof(cl_device_id) * numDevices);
    errNum = clGetDeviceIDs(
        platformIDs[0],
        CL_DEVICE_TYPE_ALL,
        numDevices, 
        &deviceIDs[0], 
        NULL);
    checkErr(errNum, "clGetDeviceIDs");

    cl_context_properties contextProperties[] =
    {
        CL_CONTEXT_PLATFORM,
        (cl_context_properties)platformIDs[0],
        0
    };

    context = clCreateContext(
        contextProperties, 
        numDevices,
        deviceIDs, 
        NULL,
        NULL, 
        &errNum);
    checkErr(errNum, "clCreateContext");

    // Create program from source
    program = clCreateProgramWithSource(
        context, 
        1, 
        &src, 
        &length, 
        &errNum);
    checkErr(errNum, "clCreateProgramWithSource");

    // Build program
    errNum = clBuildProgram(
        program,
        numDevices,
        deviceIDs,
        "-I.",
        NULL,
        NULL);
    if (errNum != CL_SUCCESS) 
    {
        // Determine the reason for the error
        char buildLog[16384];
        clGetProgramBuildInfo(
            program, 
            deviceIDs[0], 
            CL_PROGRAM_BUILD_LOG,
            sizeof(buildLog), 
            buildLog, 
            NULL);

            std::cerr << "Error in OpenCL C source: " << std::endl;
            std::cerr << buildLog;
            checkErr(errNum, "clBuildProgram");
    }

    // create buffers and sub-buffers
    inputOutput = new int[NUM_BUFFER_ELEMENTS];
    for (unsigned int i = 0; i < NUM_BUFFER_ELEMENTS; i++)
    {
        inputOutput[i] = i;
    }

    // create a single buffer to cover all the input data
    cl_mem main_buffer = clCreateBuffer(
        context,
        CL_MEM_READ_WRITE,
        sizeof(int) * NUM_BUFFER_ELEMENTS,
        NULL,
        &errNum);
    checkErr(errNum, "clCreateBuffer");

    // now for all devices other than the first create a sub-buffer
    const unsigned int numSubBuffers = NUM_BUFFER_ELEMENTS / SUB_BUFFER_SIZE;
    for (unsigned int i = 0; i < numSubBuffers; i++)
    {
        cl_buffer_region region = 
            {
                SUB_BUFFER_SIZE * i * sizeof(int),
                SUB_BUFFER_SIZE * sizeof(int)
            };
        cl_mem buffer = clCreateSubBuffer(
            main_buffer,
            CL_MEM_READ_WRITE,
            CL_BUFFER_CREATE_TYPE_REGION,
            &region,
            &errNum);
        checkErr(errNum, "clCreateSubBuffer");

        buffers.push_back(buffer);
    }

    // Create command queues
    for (unsigned int i = 0; i < numSubBuffers; i++)
    {
        const unsigned int deviceIdx = i % numDevices;

        InfoDevice<cl_device_type>::display(
            deviceIDs[deviceIdx],
            CL_DEVICE_TYPE, 
            "CL_DEVICE_TYPE");

        cl_command_queue queue = 
            clCreateCommandQueue(
                context,
                deviceIDs[deviceIdx],
                CL_QUEUE_PROFILING_ENABLE,
                &errNum);
        checkErr(errNum, "clCreateCommandQueue");

        queues.push_back(queue);

        cl_kernel kernel = clCreateKernel(
            program,
            "square",
            &errNum);
        checkErr(errNum, "clCreateKernel(square)");

        errNum = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&buffers[i]);
        checkErr(errNum, "clSetKernelArg(square)");
        errNum = clSetKernelArg(kernel, 1, sizeof(cl_uint), (void*)&numSubBuffers);
        checkErr(errNum, "clSetKernelArg(square)");

        kernels.push_back(kernel);
    }

    // Write input data
    errNum = clEnqueueWriteBuffer(
        queues[numSubBuffers - 1],
        main_buffer,
        CL_TRUE,
        0,
        sizeof(int) * NUM_BUFFER_ELEMENTS,
        (void*)inputOutput,
        0,
        NULL,
        NULL);

    std::vector<cl_event> events;
    // call kernel for each device
    for (unsigned int i = 0; i < queues.size(); i++)
    {
        cl_event event;

        size_t gWI = 1;  // for loop in kernel to average

        errNum = clEnqueueNDRangeKernel(
            queues[i], 
            kernels[i], 
            1, 
            NULL,
            (const size_t*)&gWI, 
            (const size_t*)NULL, 
            0, 
            0, 
            &event);

        events.push_back(event);
    }

    // Technically don't need this as we are doing a blocking read
    // with in-order queue.
    clWaitForEvents(events.size(), &events[0]);
    for (unsigned int i = 0; i < events.size(); i++)
    {
        cl_ulong time_start;
        cl_ulong time_end;
        clGetEventProfilingInfo(events[i], CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
        clGetEventProfilingInfo(events[i], CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);
        double nanoSeconds = time_end - time_start;
        printf("Sub buffer %d: %0.3f milliseconds \n", i, nanoSeconds / 1000000.0);
    }

    // Read back computed data
    clEnqueueReadBuffer(
        queues[numSubBuffers - 1],
        main_buffer,
        CL_TRUE,
        0,
        sizeof(int) * NUM_BUFFER_ELEMENTS,
        (void*)inputOutput,
        0,
        NULL,
        NULL);

    // Display output in rows
    for (unsigned i = 0; i < numSubBuffers; i++)
    {
        for (unsigned elems = i * SUB_BUFFER_SIZE; elems < ((i+1) * SUB_BUFFER_SIZE); elems++)
        {
            std::cout << " " << inputOutput[elems];
        }

        std::cout << std::endl;
    }

    std::cout << "Program completed successfully" << std::endl;

    return 0;
}
