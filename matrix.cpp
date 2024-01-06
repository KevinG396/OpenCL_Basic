#include <CL/cl.h>
#include <iostream>
#include <vector>
#include <stdexcept>
#include <chrono> // time

// OpenCL matrix mul kernel
const char *kernelSource = 
R"(
__kernel void matrixMul(__global float* A, __global float* B, __global float* C, const unsigned int N) {
    int row = get_global_id(0);
    int col = get_global_id(1);
    float sum = 0.0f;
    if(row < N && col < N) {
        for(int k = 0; k < N; k++) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}
)";


int main() {
    unsigned int N;
    std::cout << "Enter the size of the matrices: ";
    std::cin >> N;

    std::vector<float> A(N * N);
    std::vector<float> B(N * N);
    std::vector<float> C(N * N);

    std::cout << "Enter elements for matrix A:" << std::endl;
    for (unsigned int i = 0; i < N * N; i++) {
        std::cin >> A[i];
    }

    std::cout << "Enter elements for matrix B:" << std::endl;
    for (unsigned int i = 0; i < N * N; i++) {
        std::cin >> B[i];
    }

    try {
        cl_platform_id cpPlatform;
        cl_device_id device_id;
        cl_context context;
        cl_command_queue queue;
        cl_program program;
        cl_kernel kernel;

        // get platform and device
        clGetPlatformIDs(1, &cpPlatform, nullptr); 
        // 1: num of platform
        clGetDeviceIDs(cpPlatform, CL_DEVICE_TYPE_GPU, 1, &device_id, nullptr); 
        // platform ID; device type; num of entires(how many devices needed); 
        //device ID; num of devices found

        // create context
        context = clCreateContext(0, 1, &device_id, nullptr, nullptr, nullptr);
        // properties; device number;
        if (!context) {
            throw std::runtime_error("Failed to create a compute context");
        }

        // create command queue
        queue = clCreateCommandQueue(context, device_id, 0, nullptr);
        if (!queue) {
            throw std::runtime_error("Failed to create a command queue");
        }

        // create program
        program = clCreateProgramWithSource(context, 1, (const char **)&kernelSource, nullptr, nullptr);
        if (clBuildProgram(program, 0, nullptr, nullptr, nullptr, nullptr) != CL_SUCCESS) {
            throw std::runtime_error("Failed to build program");
        }

        // create kernel
        kernel = clCreateKernel(program, "matrixMul", nullptr);
        if (!kernel) {
            throw std::runtime_error("Failed to create kernel");
        }

        // memory allocation
        size_t bytes = N * N * sizeof(float);
        cl_mem bufA = clCreateBuffer(context, CL_MEM_READ_ONLY, bytes, nullptr, nullptr);
        cl_mem bufB = clCreateBuffer(context, CL_MEM_READ_ONLY, bytes, nullptr, nullptr);
        cl_mem bufC = clCreateBuffer(context, CL_MEM_WRITE_ONLY, bytes, nullptr, nullptr);

        // copy data from host to device
        clEnqueueWriteBuffer(queue, bufA, CL_TRUE, 0, bytes, A.data(), 0, nullptr, nullptr);
        clEnqueueWriteBuffer(queue, bufB, CL_TRUE, 0, bytes, B.data(), 0, nullptr, nullptr);

        // set kernel parameters
        clSetKernelArg(kernel, 0, sizeof(cl_mem), &bufA);
        clSetKernelArg(kernel, 1, sizeof(cl_mem), &bufB);
        clSetKernelArg(kernel, 2, sizeof(cl_mem), &bufC);
        clSetKernelArg(kernel, 3, sizeof(unsigned int), &N);

        // kernel execution
        size_t globalSize[] = {N, N};
        auto start = std::chrono::high_resolution_clock::now();  // start timing
        clEnqueueNDRangeKernel(queue, kernel, 2, nullptr, globalSize, nullptr, 0, nullptr, nullptr);
        // command queue; kernel; work_dim; offset; total work_num for each dim({N,N}); 
        // num_work for each group; num_events_in_wait_list; *event_wait_list; *event(new event obj)
        clFinish(queue);
        auto end = std::chrono::high_resolution_clock::now(); // finish timing

        // host read result from device (copy data to host))
        clEnqueueReadBuffer(queue, bufC, CL_TRUE, 0, bytes, C.data(), 0, nullptr, nullptr);

        // print result matrix C
        std::cout << "Result Matrix C:" << std::endl;
        for (unsigned int i = 0; i < N; ++i) {
            for (unsigned int j = 0; j < N; ++j) {
                std::cout << C[i * N + j] << " ";
            }
            std::cout << std::endl;
        }

        std::chrono::duration<double, std::milli> exec_time = end - start;
        std::cout << "Execution time: " << exec_time.count() << " ms" << std::endl;


        // release resource
        clReleaseMemObject(bufA);
        clReleaseMemObject(bufB);
        clReleaseMemObject(bufC);
        clReleaseProgram(program);
        clReleaseKernel(kernel);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
