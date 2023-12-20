// Includes
#include <stdio.h>
#include <iostream>
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#include <CL/cl.h>
#include "MATRIX.h"

// =================================================================================================

// Repeat all kernels multiple times to get an average timing result
#define NUM_RUNS 2

// Size of the matrices - K, M, N (squared)
#define SIZE 4096

// Threadblock sizes (e.g. for kernels myGEMM1 or myGEMM2)
#define TS 32

const int m_size = 64;

// =================================================================================================

// Set the kernel as a string (better to do this in a separate file though)
const char* kernelstring =
"__kernel void myGEMM1(const int S,"
"                      const __global uint32_t* A,"
"                      const __global uint32_t* B,"
"                      __global uint32_t* C) {"
"    const int globalRow = get_global_id(0);"
"    const int globalCol = get_global_id(1);"
"    uint32_t buf = 0;"
"    for (int i = 0; i < S; i++) {"
"        acc += A[i * S + globalRow] * B[globalCol * S + i];"
"    }"
"    C[globalCol * S + globalRow] = buf;"
"}";
//"__kernel void myGEMM1(const int m_size,"
//"                      const __global uint32_t* A,"
//"                      const __global uint32_t* B,"
//"                      __global uint32_t* C) {"
//"    const int globalRow = get_global_id(0);"
//"    const int globalCol = get_global_id(1);"
//"    uint32_t buf = 0;"
//"    for (int k=0; k<K; k++) {"
//"        acc += A[k*M + globalRow] * B[globalCol*K + k];"
//"    }"
//"    C[globalCol*M + globalRow] = acc;"
//"}";

// =================================================================================================

// Matrix-multiplication using a custom OpenCL SGEMM kernel.
int main(int argc, char* argv[]) {

    // Set the sizes
    int K = SIZE;
    int M = SIZE;
    int N = SIZE;

    // Create the matrices and initialize them with random values
    uint32_t mat_a[m_size * m_size];
    uint32_t mat_b[m_size * m_size];
    uint32_t mat_scalar[m_size * m_size];
    uint32_t mat_cl[m_size * m_size];

    MAT_fill_random(mat_a, m_size);
    MAT_fill_random(mat_b, m_size);
    MAT_fill_empty(mat_scalar, m_size);
    MAT_fill_empty(mat_cl, m_size);

    // Configure the OpenCL environment
    printf(">>> Initializing OpenCL...\n");
    cl_platform_id platform = 0;
    clGetPlatformIDs(1, &platform, NULL);
    cl_device_id device = 0;
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, NULL);
    cl_command_queue queue = clCreateCommandQueue(context, device, 0, NULL);
    char deviceName[1024];
    clGetDeviceInfo(device, CL_DEVICE_NAME, 1024, deviceName, NULL);
    cl_event event = NULL;

    // Compile the kernel
    cl_program program = clCreateProgramWithSource(context, 1, &kernelstring, NULL, NULL);
    clBuildProgram(program, 0, NULL, "", NULL, NULL);

    // Prepare OpenCL memory objects
    int buffer_size = m_size * m_size * sizeof(uint32_t);
    cl_mem bufA = clCreateBuffer(context, CL_MEM_READ_ONLY, buffer_size, NULL, NULL);
    cl_mem bufB = clCreateBuffer(context, CL_MEM_READ_ONLY, buffer_size, NULL, NULL);
    cl_mem bufC = clCreateBuffer(context, CL_MEM_READ_WRITE, buffer_size, NULL, NULL);

    // Copy matrices to the GPU
    clEnqueueWriteBuffer(queue, bufA, CL_TRUE, 0, buffer_size, mat_a, 0, NULL, NULL);
    clEnqueueWriteBuffer(queue, bufB, CL_TRUE, 0, buffer_size, mat_b, 0, NULL, NULL);
    clEnqueueWriteBuffer(queue, bufC, CL_TRUE, 0, buffer_size, mat_cl, 0, NULL, NULL);

    // Configure the myGEMM kernel and set its arguments
    cl_kernel kernel = clCreateKernel(program, "myGEMM1", NULL);
    clSetKernelArg(kernel, 0, sizeof(int), (void*)&m_size);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&bufA);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*)&bufB);
    clSetKernelArg(kernel, 3, sizeof(cl_mem), (void*)&bufC);

    // Start the timed loop
    printf(">>> Starting %d myGEMM runs...\n", NUM_RUNS);
    for (int r = 0; r < NUM_RUNS; r++) {

        // Run the myGEMM kernel
        const size_t local[2] = { TS, TS };
        const size_t global[2] = { m_size, m_size };
        clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global, local, 0, NULL, &event);

        // Wait for calculations to be finished
        clWaitForEvents(1, &event);
    }

    // Copy the output matrix C back to the CPU memory
    clEnqueueReadBuffer(queue, bufC, CL_TRUE, 0, buffer_size, mat_cl, 0, NULL, NULL);

    // Free the OpenCL memory objects
    clReleaseMemObject(bufA);
    clReleaseMemObject(bufB);
    clReleaseMemObject(bufC);

    // Clean-up OpenCL 
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    clReleaseProgram(program);
    clReleaseKernel(kernel);

    // Exit
    return 0;
}