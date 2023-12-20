// Vladimir Klyzhko
// 
// OpenCL with OpenCV
//
// Processing images with gaussian filter on CPU and GPU
// Source images are stored in ../images/source
// Result images are stored in ../images/result

#include <iostream>
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/utils/logger.hpp>
#include <chrono>
#include <CL/cl.hpp>
#include <fstream>

using namespace std;
using namespace cv;

string path_source = "images/source/";
string path_result = "images/result/";
// need custom t_size for 8k.png
vector<string> images{ "mountain.png", "city.png", "desert.png"/*, "8k.png" */}; 

const int t_size = 32; //thread block size
const int test_count = 10;

const int g_size = 3;
const float gaussian_filter[g_size * g_size] = {
    1.0 / 16.0, 2.0 / 16.0, 1.0 / 16.0,
    2.0 / 16.0, 4.0 / 16.0, 2.0 / 16.0,
    1.0 / 16.0, 2.0 / 16.0, 1.0 / 16.0
};

void CPU_Processing(Mat& input, Mat& output) {
    for (int i = 0; i < input.rows; i++) {
        for (int j = 0; j < input.cols; j++) {
            float blur_pixel = 0.0;
            for (int x = -g_size / 2; x <= g_size / 2; x++) {
                for (int y = -g_size / 2; y <= g_size / 2; y++) {
                    int current_row = i + x;
                    int current_col = j + y;

                    if (current_row >= 0 && current_row < input.rows && current_col >= 0 && current_col < input.cols) {
                        float filter_value = gaussian_filter[(y + g_size / 2) * g_size + (x + g_size / 2)];
                        blur_pixel += input.at<uchar>(i, j) * filter_value;
                    }
                }
            }
            output.at<uchar>(i, j) = static_cast <uchar>(blur_pixel);
        }
    }
};

int main() {
    // mute opencv log
    utils::logging::setLogLevel(utils::logging::LogLevel::LOG_LEVEL_SILENT);

    chrono::time_point<chrono::system_clock> t_start;
    chrono::time_point<chrono::system_clock> t_end;

    double t_cpu_avg = 0;
    double t_opencl_avg = 0;

    // pass file to string
    std::string kernel_file = "gauss_filter.cl";
    std::ifstream ifs{ kernel_file };
    if (!ifs) {
        std::cout << "Cannot open file " << kernel_file << "\n";
        return -1;
    }
    std::string kernel_code((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());

    // get all platforms (platforms = drivers for each technology, i.e. amd, nvidia or intel)
    std::vector<cl::Platform> all_platforms;
    cl::Platform::get(&all_platforms);
    if (all_platforms.size() == 0) {
        std::cout << " No platforms found. Check OpenCL installation!\n";
        exit(1);
    }
    cl::Platform default_platform = all_platforms[0];
    std::cout << "Using platform: " << default_platform.getInfo<CL_PLATFORM_NAME>() << "\n";

    // get default device of the default platform
    std::vector<cl::Device> all_devices;
    default_platform.getDevices(CL_DEVICE_TYPE_ALL, &all_devices);
    if (all_devices.size() == 0) {
        std::cout << " No devices found. Check OpenCL installation!\n";
        exit(1);
    }
    cl::Device default_device = all_devices[0];
    std::cout << "Using device: " << default_device.getInfo<CL_DEVICE_NAME>() << "\n";

    // context is managing command queues, memory, program and kernel objects on devices inside this context
    cl::Context context({ default_device });

    //std::vector<cl::Device> devices = context.getInfo<CL_CONTEXT_DEVICES>();
    //int num;
    //devices[0].getInfo(CL_DEVICE_MAX_COMPUTE_UNITS, &num);
    //cout << "\nMultiprocessors: " << num << "\n\n";

    // object containing kernel code
    cl::Program::Sources sources;

    sources.push_back({ kernel_code.c_str(),kernel_code.length() });

    cl::Program program(context, sources);
    if (program.build({ default_device }) != CL_SUCCESS) {
        std::cout << " Error building: " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(default_device) << "\n";
        exit(1);
    }

    auto iter{ images.begin() };
    while (iter != images.end()) {
        Mat input = imread(path_source + *iter);
        if (input.empty()) {
            cout << "Image error\n";
            return 0;
        }

        int rows = input.rows;
        int cols = input.cols;

        cout << "Current image: " << *iter << " (" << input.cols << "x" << input.rows << ")\n";

        uchar* mat_input, * mat_output;

        Mat gray_input;
        cvtColor(input, gray_input, COLOR_BGR2GRAY);

        // create buffers on the device
        int image_size = input.size().area() * sizeof(char);
        int filter_size = g_size * g_size * sizeof(float);
        cl::Buffer g_filter(context, CL_MEM_READ_WRITE, filter_size);
        cl::Buffer image_in(context, CL_MEM_READ_WRITE, image_size);
        cl::Buffer image_out(context, CL_MEM_READ_WRITE, image_size);

        //create queue to which we will push commands for the device.
        cl::CommandQueue queue(context, default_device);

        queue.enqueueWriteBuffer(g_filter, CL_TRUE, 0, filter_size, gaussian_filter);
        queue.enqueueWriteBuffer(image_in, CL_TRUE, 0, image_size, gray_input.data);

        cl::Kernel kernel_add = cl::Kernel(program, "opencl_processing");
        kernel_add.setArg(0, g_filter);
        kernel_add.setArg(1, image_in);
        kernel_add.setArg(2, image_out);
        kernel_add.setArg(3, input.rows);
        kernel_add.setArg(4, input.cols);

        // CPU test
        for (int test = 0; test < test_count; test++) {
            Mat cpu_output = gray_input.clone();
            t_start = chrono::system_clock::now();
            CPU_Processing(gray_input, cpu_output);
            t_end = chrono::system_clock::now();
            imwrite(path_result + "CPU_" + *iter, cpu_output);
            double t_cpu = chrono::duration_cast<chrono::nanoseconds>(t_end - t_start).count() / 1000000.0;
            t_cpu_avg += t_cpu;
        }
        t_cpu_avg /= test_count;
        cout << "Average CPU time: " << t_cpu_avg << " milliseconds\n";

        // GPU test
        for (int test = 0; test < test_count; test++) {
            Mat gpu_output = gray_input.clone();
            t_start = chrono::system_clock::now();
            queue.enqueueNDRangeKernel(kernel_add, cl::NullRange, cl::NDRange(input.rows, input.cols), cl::NDRange(16, 64));
            queue.finish();
            queue.enqueueReadBuffer(image_out, CL_TRUE, 0, image_size, gpu_output.data);
            t_end = chrono::system_clock::now();
            imwrite(path_result + "GPU_" + *iter, gpu_output);
            double t_gpu = chrono::duration_cast<chrono::nanoseconds>(t_end - t_start).count() / 1000000.0;
            t_opencl_avg += t_gpu;
        }
        t_opencl_avg /= test_count;
        cout << "Average GPU time: " << t_opencl_avg << " milliseconds\n";
        cout << "CPU time / GPU time = " << t_cpu_avg / t_opencl_avg << "\n";

        cout << "\n";

        ++iter;
    }

    return 0;
}
