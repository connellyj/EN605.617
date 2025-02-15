//
// Book:      OpenCL(R) Programming Guide
// Authors:   Aaftab Munshi, Benedict Gaster, Timothy Mattson, James Fung, Dan Ginsburg
// ISBN-10:   0-321-74964-2
// ISBN-13:   978-0-321-74964-2
// Publisher: Addison-Wesley Professional
// URLs:      http://safari.informit.com/9780132488006/
//            http://www.openclprogrammingguide.com
//

// HelloWorld.cpp
//
//    This is a simple example that demonstrates basic OpenCL setup and
//    use.

// Modified by Julia Connelly

#include <iostream>
#include <fstream>
#include <sstream>

#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

///
//  Constants
//
const int ARRAY_SIZE = 2 << 16;

///
//  Create an OpenCL context on the first available platform using
//  either a GPU or CPU depending on what is available.
//
cl_context CreateContext()
{
    cl_int errNum;
    cl_uint numPlatforms;
    cl_platform_id firstPlatformId;
    cl_context context = NULL;

    // First, select an OpenCL platform to run on.  For this example, we
    // simply choose the first available platform.  Normally, you would
    // query for all available platforms and select the most appropriate one.
    errNum = clGetPlatformIDs(1, &firstPlatformId, &numPlatforms);
    if (errNum != CL_SUCCESS || numPlatforms <= 0)
    {
        printf("Failed to find any OpenCL platforms.\n");
        return NULL;
    }

    // Next, create an OpenCL context on the platform.  Attempt to
    // create a GPU-based context, and if that fails, try to create
    // a CPU-based context.
    cl_context_properties contextProperties[] =
    {
        CL_CONTEXT_PLATFORM,
        (cl_context_properties)firstPlatformId,
        0
    };
    context = clCreateContextFromType(contextProperties, CL_DEVICE_TYPE_GPU,
                                      NULL, NULL, &errNum);
    if (errNum != CL_SUCCESS)
    {
        printf("Could not create GPU context, trying CPU...\n");
        context = clCreateContextFromType(contextProperties, CL_DEVICE_TYPE_CPU,
                                          NULL, NULL, &errNum);
        if (errNum != CL_SUCCESS)
        {
            printf("Failed to create an OpenCL GPU or CPU context.\n");
            return NULL;
        }
    }

    return context;
}

///
//  Create a command queue on the first device available on the
//  context
//
cl_command_queue CreateCommandQueue(cl_context context, cl_device_id *device)
{
    cl_int errNum;
    cl_device_id *devices;
    cl_command_queue commandQueue = NULL;
    size_t deviceBufferSize = -1;

    // First get the size of the devices buffer
    errNum = clGetContextInfo(context, CL_CONTEXT_DEVICES, 0, NULL, &deviceBufferSize);
    if (errNum != CL_SUCCESS)
    {
        printf("Failed call to clGetContextInfo(...,GL_CONTEXT_DEVICES,...)\n");
        return NULL;
    }

    if (deviceBufferSize <= 0)
    {
        printf("No devices available.\n");
        return NULL;
    }

    // Allocate memory for the devices buffer
    devices = new cl_device_id[deviceBufferSize / sizeof(cl_device_id)];
    errNum = clGetContextInfo(context, CL_CONTEXT_DEVICES, deviceBufferSize, devices, NULL);
    if (errNum != CL_SUCCESS)
    {
        delete [] devices;
        printf("Failed to get device IDs\n");
        return NULL;
    }

    // In this example, we just choose the first available device.  In a
    // real program, you would likely use all available devices or choose
    // the highest performance device based on OpenCL device queries
    commandQueue = clCreateCommandQueue(context, devices[0], CL_QUEUE_PROFILING_ENABLE, NULL);
    if (commandQueue == NULL)
    {
        delete [] devices;
        printf("Failed to create commandQueue for device 0\n");
        return NULL;
    }

    *device = devices[0];
    delete [] devices;
    return commandQueue;
}

///
//  Create an OpenCL program from the kernel source file
//
cl_program CreateProgram(cl_context context, cl_device_id device, const char* fileName)
{
    cl_int errNum;
    cl_program program;

    std::ifstream kernelFile(fileName, std::ios::in);
    if (!kernelFile.is_open())
    {
        printf("Failed to open file for reading: %s\n", fileName);
        return NULL;
    }

    std::ostringstream oss;
    oss << kernelFile.rdbuf();

    std::string srcStdStr = oss.str();
    const char *srcStr = srcStdStr.c_str();
    program = clCreateProgramWithSource(context, 1,
                                        (const char**)&srcStr,
                                        NULL, NULL);
    if (program == NULL)
    {
        printf("Failed to create CL program from source.\n");
        return NULL;
    }

    errNum = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    if (errNum != CL_SUCCESS)
    {
        // Determine the reason for the error
        char buildLog[16384];
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG,
                              sizeof(buildLog), buildLog, NULL);

        printf("Error in kernel\n");
        clReleaseProgram(program);
        return NULL;
    }

    return program;
}

///
//  Create memory objects used as the arguments to the kernel
//  The kernel takes three arguments: result (output), a (input),
//  and b (input)
//
bool CreateMemObjects(cl_context context, cl_mem memObjects[3],
                      float *a, float *b)
{
    memObjects[0] = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                   sizeof(float) * ARRAY_SIZE, a, NULL);
    memObjects[1] = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                   sizeof(float) * ARRAY_SIZE, b, NULL);
    memObjects[2] = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                   sizeof(float) * ARRAY_SIZE, NULL, NULL);

    if (memObjects[0] == NULL || memObjects[1] == NULL || memObjects[2] == NULL)
    {
        printf("Error creating memory objects.\n");
        return false;
    }

    return true;
}

///
//  Cleanup any created OpenCL resources
//
void Cleanup(cl_context context, cl_command_queue commandQueue,
             cl_program program, cl_mem memObjects[3])
{
    for (int i = 0; i < 3; i++)
    {
        if (memObjects[i] != 0)
            clReleaseMemObject(memObjects[i]);
    }
    if (commandQueue != 0)
        clReleaseCommandQueue(commandQueue);

    if (program != 0)
        clReleaseProgram(program);

    if (context != 0)
        clReleaseContext(context);

}

int ExecuteKernel(const cl_program program, const cl_command_queue commandQueue, const char* kernelName, const cl_mem* memObjects)
{
    // Create OpenCL kernel
    cl_kernel kernel = clCreateKernel(program, kernelName, NULL);
    if (kernel == NULL)
    {
        printf("Failed to create kernel: %s\n", kernelName);
        return 1;
    }

    // Set the kernel arguments (result, a, b)
    cl_int errNum;
    errNum = clSetKernelArg(kernel, 0, sizeof(cl_mem), &memObjects[0]);
    errNum |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &memObjects[1]);
    errNum |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &memObjects[2]);
    if (errNum != CL_SUCCESS)
    {
        printf("Error setting kernel arguments: %s\n", kernelName);
        clReleaseKernel(kernel);
        return 1;
    }

    size_t globalWorkSize[1] = { ARRAY_SIZE };
    size_t localWorkSize[1] = { 1 };

    // Queue the kernel up for execution across the array
    cl_event event;
    errNum = clEnqueueNDRangeKernel(commandQueue, kernel, 1, NULL, globalWorkSize, localWorkSize, 0, NULL, &event);
    if (errNum != CL_SUCCESS)
    {
        printf("Error queuing kernel for execution: %s\n", kernelName);
        clReleaseKernel(kernel);
        return 1;
    }

    // Read the output buffer back to the Host
    float result[ARRAY_SIZE];
    errNum = clEnqueueReadBuffer(commandQueue, memObjects[2], CL_TRUE,
        0, ARRAY_SIZE * sizeof(float), result,
        0, NULL, NULL);
    if (errNum != CL_SUCCESS)
    {
        printf("Error reading result buffer: %s\n", kernelName);
        clReleaseKernel(kernel);
        return 1;
    }

    // Output the result buffer
    for (int i = 0; i < 10; i++)
    {
        printf("%f ", result[i]);
    }
    printf("...\n");

    clWaitForEvents(1, &event);
    cl_ulong time_start;
    cl_ulong time_end;
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);
    double nanoSeconds = time_end - time_start;
    printf("OpenCl Execution time for '%s' is: %0.3f milliseconds \n", kernelName, nanoSeconds / 1000000.0);

    return 0;
}

///
//	main() for HelloWorld example
//
int main(int argc, char** argv)
{
    cl_context context = 0;
    cl_command_queue commandQueue = 0;
    cl_program program = 0;
    cl_device_id device = 0;
    cl_mem memObjects[3] = { 0, 0, 0 };
    cl_int errNum;

    // Create an OpenCL context on first available platform
    context = CreateContext();
    if (context == NULL)
    {
        printf("Failed to create OpenCL context.\n");
        return 1;
    }

    // Create a command-queue on the first device available
    // on the created context
    commandQueue = CreateCommandQueue(context, &device);
    if (commandQueue == NULL)
    {
        Cleanup(context, commandQueue, program, memObjects);
        return 1;
    }

    // Create OpenCL program from HelloWorld.cl kernel source
    program = CreateProgram(context, device, "assignment.cl");
    if (program == NULL)
    {
        Cleanup(context, commandQueue, program, memObjects);
        return 1;
    }

    // Create memory objects that will be used as arguments to
    // kernel.  First create host memory arrays that will be
    // used to store the arguments to the kernel
    float a[ARRAY_SIZE];
    float b[ARRAY_SIZE];
    for (int i = 0; i < ARRAY_SIZE; i++)
    {
        a[i] = (float)i + 1;
        b[i] = (float)((i + 1) * 2);
    }

    if (!CreateMemObjects(context, memObjects, a, b))
    {
        Cleanup(context, commandQueue, program, memObjects);
        return 1;
    }

    // Execute all math kernels
    if (ExecuteKernel(program, commandQueue, "add", memObjects) != 0)
    {
        Cleanup(context, commandQueue, program, memObjects);
        return 1;
    }
    if (ExecuteKernel(program, commandQueue, "sub", memObjects) != 0)
    {
        Cleanup(context, commandQueue, program, memObjects);
        return 1;
    }
    if (ExecuteKernel(program, commandQueue, "mult", memObjects) != 0)
    {
        Cleanup(context, commandQueue, program, memObjects);
        return 1;
    }
    if (ExecuteKernel(program, commandQueue, "div", memObjects) != 0)
    {
        Cleanup(context, commandQueue, program, memObjects);
        return 1;
    }
    if (ExecuteKernel(program, commandQueue, "power", memObjects) != 0)
    {
        Cleanup(context, commandQueue, program, memObjects);
        return 1;
    }

    Cleanup(context, commandQueue, program, memObjects);

    return 0;
}
