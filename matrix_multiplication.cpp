// Vladimir Klyzhko
// 
// Matrix multiplication with OpenCL

#include <iostream>
#include <stdio.h>
#include <stdint.h>
#include <chrono>
#include <ctime>
#include "MATRIX.h"
#include <CL/cl.hpp>
#include <fstream>

using namespace std;

const int m_size = 1024;
const int t_size = 32; //thread block size
const int test_count = 1;

uint32_t mat_a[m_size * m_size];
uint32_t mat_b[m_size * m_size];
uint32_t mat_SCALAR[m_size * m_size];
uint32_t mat_CL[m_size * m_size];

int main() {
    chrono::time_point<chrono::system_clock> t_start;
    chrono::time_point<chrono::system_clock> t_end;

    std::string kernel_file = "matrix_kernel.cl";
    std::ifstream ifs{ kernel_file };
    if (!ifs) {
        std::cout << "Cannot open file " << kernel_file << "\n";
        return -1;
    }
    std::string kernel_code((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
    std::cout << kernel_code << '\n';

    MAT_fill_random(mat_a, m_size);
    MAT_fill_random(mat_b, m_size);
    MAT_fill_empty(mat_SCALAR, m_size);
    MAT_fill_empty(mat_CL, m_size);

    printf("Matrix size: %dx%d\n", m_size, m_size);

    //get all platforms (drivers)
    std::vector<cl::Platform> all_platforms;
    cl::Platform::get(&all_platforms);
    if (all_platforms.size() == 0) {
        std::cout << " No platforms found. Check OpenCL installation!\n";
        exit(1);
    }
    cl::Platform default_platform = all_platforms[0];
    std::cout << "Using platform: " << default_platform.getInfo<CL_PLATFORM_NAME>() << "\n";

    //get default device of the default platform
    std::vector<cl::Device> all_devices;
    default_platform.getDevices(CL_DEVICE_TYPE_ALL, &all_devices);
    if (all_devices.size() == 0) {
        std::cout << " No devices found. Check OpenCL installation!\n";
        exit(1);
    }
    cl::Device default_device = all_devices[0];
    std::cout << "Using device: " << default_device.getInfo<CL_DEVICE_NAME>() << "\n";

    cl::Context context({ default_device });
    std::vector<cl::Device> devices = context.getInfo<CL_CONTEXT_DEVICES>();

    //int num;
    //devices[0].getInfo(CL_DEVICE_MAX_COMPUTE_UNITS, &num);
    //cout << "\nMultiprocessors: " << num << "\n\n";

    cl::Program::Sources sources;

    sources.push_back({ kernel_code.c_str(),kernel_code.length() });

    cl::Program program(context, sources);
    if (program.build({ default_device }) != CL_SUCCESS) {
        std::cout << " Error building: " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(default_device) << "\n";
        exit(1);
    }

    // create buffers on the device
    cl_int buffer_size = m_size * m_size * sizeof(cl_uint);
    cl::Buffer buffer_A(context, CL_MEM_READ_WRITE, buffer_size);
    cl::Buffer buffer_B(context, CL_MEM_READ_WRITE, buffer_size);
    cl::Buffer buffer_C(context, CL_MEM_READ_WRITE, buffer_size);

    //create queue to which we will push commands for the device.
    cl::CommandQueue queue(context, default_device);

    queue.enqueueWriteBuffer(buffer_A, CL_TRUE, 0, buffer_size, mat_a);
    queue.enqueueWriteBuffer(buffer_B, CL_TRUE, 0, buffer_size, mat_b);

    cl::Kernel kernel_add = cl::Kernel(program, "matrix_mult");
    kernel_add.setArg(0, buffer_A);
    kernel_add.setArg(1, buffer_B);
    kernel_add.setArg(2, buffer_C);
    kernel_add.setArg(3, m_size);

    double avg_time = 0;
    for (int test = 0; test < test_count; test++) {
        t_start = chrono::system_clock::now();
        queue.enqueueNDRangeKernel(kernel_add, cl::NullRange, cl::NDRange(m_size, m_size), cl::NDRange(t_size, t_size));
        queue.finish();
        queue.enqueueReadBuffer(buffer_C, CL_TRUE, 0, buffer_size, mat_CL);
        t_end = chrono::system_clock::now();

        double cur_time = chrono::duration_cast<chrono::milliseconds>(t_end - t_start).count() / 1000.0;
        avg_time += cur_time;

        cout << "Attempt " << test + 1 << ", time: " << cur_time << " seconds\n";
    }
    cout << "Average time is " << avg_time / test_count << " seconds\n\n";

    t_start = chrono::system_clock::now();
    MAT_scalar_multiply(mat_SCALAR, mat_a, mat_b, m_size);
    t_end = chrono::system_clock::now();
    cout << "CPU time:    " << chrono::duration_cast<chrono::milliseconds>(t_end - t_start).count() / 1000.0 << " seconds\n";

    bool equal = MAT_check_equality(mat_SCALAR, mat_CL, m_size);
    if (equal) {
        cout << "Matrices are equal\n";
    }
    else {
        cout << "Matrices aren't equal\n";
    }

    return 0;
}