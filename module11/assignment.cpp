//
// Book:      OpenCL(R) Programming Guide
// Authors:   Aaftab Munshi, Benedict Gaster, Timothy Mattson, James Fung, Dan Ginsburg
// ISBN-10:   0-321-74964-2
// ISBN-13:   978-0-321-74964-2
// Publisher: Addison-Wesley Professional
// URLs:      http://safari.informit.com/9780132488006/
//            http://www.openclprogrammingguide.com
//


// Convolution.cpp
//
//    This is a simple example that demonstrates OpenCL platform, device, and context
//    use.

// Modified by Julia Connelly
// compile windows: g++ assignment.cpp -o assignment -I "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.4\include" "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.4\lib\x64\OpenCL.lib"

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>

#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

#if !defined(CL_CALLBACK)
#define CL_CALLBACK
#endif

// Constants
const unsigned int inputSignalWidth  = 49;
const unsigned int inputSignalHeight = 49;

cl_uint inputSignal[inputSignalHeight][inputSignalWidth];

const unsigned int outputSignalWidth  = 43;
const unsigned int outputSignalHeight = 43;

float outputSignal[outputSignalHeight][outputSignalWidth];

const unsigned int maskWidth  = 7;
const unsigned int maskHeight = 7;

float mask[maskHeight][maskWidth];

const int TEST_SIGNAL = 0;
const int NUM_ITER = 10;
const int NUM_OUT = 10;

///
// Function to check and handle OpenCL errors
inline void 
checkErr(cl_int err, const char * name)
{
    if (err != CL_SUCCESS) {
        std::cerr << "ERROR: " <<  name << " (" << err << ")" << std::endl;
        exit(EXIT_FAILURE);
    }
}

void CL_CALLBACK contextCallback(
	const char * errInfo,
	const void * private_info,
	size_t cb,
	void * user_data)
{
	std::cout << "Error occured during context use: " << errInfo << std::endl;
	// should really perform any clearup and so on at this point
	// but for simplicitly just exit.
	exit(1);
}

void fillInput()
{
	int w, h;
	for (w = 0; w < inputSignalWidth; ++w)
	{
		for (h = 0; h < inputSignalHeight; ++h)
		{
			if (TEST_SIGNAL)
			{
				inputSignal[w][h] = 1;
			}
			else
			{
				inputSignal[w][h] = rand() % 9;
			}
		}
	}
}

void fillMask()
{
	int w, h;
	const int center = (maskWidth - 1) / 2;
	for (w = 0; w < maskWidth; ++w)
	{
		for (h = 0; h < maskHeight; ++h)
		{
			const int ww = abs(center - w);
			const int hh = abs(center - h);
			const int dist = ww > hh ? ww : hh;
			mask[w][h] = (center - dist + 1.0f) / (center + 1.0f);
		}
	}
}

void executeKernel(cl_context context, cl_command_queue queue, cl_kernel kernel, cl_mem outputSignalBuffer)
{
	fillInput();

	cl_int errNum;
	cl_mem inputSignalBuffer = clCreateBuffer(
		context,
		CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		sizeof(cl_uint) * inputSignalHeight * inputSignalWidth,
		static_cast<void*>(inputSignal),
		&errNum);
	checkErr(errNum, "clCreateBuffer(inputSignal)");

	errNum = clSetKernelArg(kernel, 0, sizeof(cl_mem), &inputSignalBuffer);

	const size_t globalWorkSize[2] = { outputSignalWidth, outputSignalHeight };
	const size_t localWorkSize[2] = { 1, 1 };

	// Queue the kernel up for execution across the array
	cl_event event;
	errNum = clEnqueueNDRangeKernel(
		queue,
		kernel,
		2,
		NULL,
		globalWorkSize,
		localWorkSize,
		0,
		NULL,
		&event);
	checkErr(errNum, "clEnqueueNDRangeKernel");

	clWaitForEvents(1, &event);
	cl_ulong time_start;
	cl_ulong time_end;
	clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
	clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);
	double nanoSeconds = time_end - time_start;
	printf("OpenCl Execution time is: %0.3f milliseconds \n", nanoSeconds / 1000000.0);
	std::cout << std::endl;

	errNum = clEnqueueReadBuffer(
		queue,
		outputSignalBuffer,
		CL_TRUE,
		0,
		sizeof(float) * outputSignalHeight * outputSignalHeight,
		outputSignal,
		0,
		NULL,
		NULL);
	checkErr(errNum, "clEnqueueReadBuffer");

	// Output the result buffer
	for (int y = 0; y < NUM_OUT; y++)
	{
		for (int x = 0; x < NUM_OUT; x++)
		{
			printf("%.2f ", outputSignal[y][x]);
		}
		std::cout << " ..." << std::endl;
	}
	std::cout << "..." << std::endl;
	std::cout << std::endl;
}

///
//	main() for Convoloution example
//
int main(int argc, char** argv)
{
    cl_int errNum;
    cl_uint numPlatforms;
	cl_uint numDevices;
    cl_platform_id * platformIDs;
	cl_device_id * deviceIDs;
    cl_context context = NULL;
	cl_command_queue queue;
	cl_program program;
	cl_kernel kernel;
	cl_mem outputSignalBuffer;
	cl_mem maskBuffer;

	fillMask();

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

	// Iterate through the list of platforms until we find one that supports
	// a CPU device, otherwise fail with an error.
	deviceIDs = NULL;
	cl_uint i;
	for (i = 0; i < numPlatforms; i++)
	{
		errNum = clGetDeviceIDs(
            platformIDs[i], 
            CL_DEVICE_TYPE_GPU, 
            0,
            NULL,
            &numDevices);
		if (errNum != CL_SUCCESS && errNum != CL_DEVICE_NOT_FOUND)
	    {
			checkErr(errNum, "clGetDeviceIDs");
        }
	    else if (numDevices > 0) 
	    {
		   	deviceIDs = (cl_device_id *)alloca(sizeof(cl_device_id) * numDevices);
			errNum = clGetDeviceIDs(
				platformIDs[i],
				CL_DEVICE_TYPE_GPU,
				numDevices, 
				&deviceIDs[0], 
				NULL);
			checkErr(errNum, "clGetDeviceIDs");
			break;
	   }
	}

	// Check to see if we found at least one CPU device, otherwise return
// 	if (deviceIDs == NULL) {
// 		std::cout << "No CPU device found" << std::endl;
// 		exit(-1);
// 	}

    // Next, create an OpenCL context on the selected platform.  
    cl_context_properties contextProperties[] =
    {
        CL_CONTEXT_PLATFORM,
        (cl_context_properties)platformIDs[i],
        0
    };
    context = clCreateContext(
		contextProperties, 
		numDevices,
        deviceIDs, 
		&contextCallback,
		NULL, 
		&errNum);
	checkErr(errNum, "clCreateContext");

	std::ifstream srcFile("assignment.cl");
    checkErr(srcFile.is_open() ? CL_SUCCESS : -1, "reading assignment.cl");

	std::string srcProg(
        std::istreambuf_iterator<char>(srcFile),
        (std::istreambuf_iterator<char>()));

	const char * src = srcProg.c_str();
	size_t length = srcProg.length();

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
		NULL,
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

        std::cerr << "Error in kernel: " << std::endl;
        std::cerr << buildLog;
		checkErr(errNum, "clBuildProgram");
    }

	// Create kernel object
	kernel = clCreateKernel(
		program,
		"convolve",
		&errNum);
	checkErr(errNum, "clCreateKernel");

	// Pick the first device and create command queue.
	queue = clCreateCommandQueue(
		context,
		deviceIDs[0],
		CL_QUEUE_PROFILING_ENABLE,
		&errNum);
	checkErr(errNum, "clCreateCommandQueue");

	// Now allocate buffers
	maskBuffer = clCreateBuffer(
		context,
		CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		sizeof(float) * maskHeight * maskWidth,
		static_cast<void*>(mask),
		&errNum);
	checkErr(errNum, "clCreateBuffer(mask)");

	outputSignalBuffer = clCreateBuffer(
		context,
		CL_MEM_WRITE_ONLY,
		sizeof(float) * outputSignalHeight * outputSignalWidth,
		NULL,
		&errNum);
	checkErr(errNum, "clCreateBuffer(outputSignal)");

	errNum = clSetKernelArg(kernel, 1, sizeof(cl_mem), &maskBuffer);
	errNum |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &outputSignalBuffer);
	errNum |= clSetKernelArg(kernel, 3, sizeof(cl_uint), &inputSignalWidth);
	errNum |= clSetKernelArg(kernel, 4, sizeof(cl_uint), &maskWidth);
	checkErr(errNum, "clSetKernelArg");

	for (i = 0; i < NUM_ITER; ++i)
	{
		executeKernel(context, queue, kernel, outputSignalBuffer);
	}

	return 0;
}
