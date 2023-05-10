#include <CL/cl.h>
#include <map>
#include <string>

#include "gpu.h"
#include "../io/io.h"
#include "../utils/listWithSize.h"

using std::string;

using gpu::GpuProps;
using utils::ListWithSize;

ListWithSize<cl_device_id> *devices;
ListWithSize<cl_command_queue> *command_queues;
cl_context *context;
map<const char*, cl_kernel> *available_kernels;

#define FOR_ALL_DEVICES(stmt) for (size_t i = 0; i < devices -> num_items; i++) {stmt;}
#define SET_ERROR_IF_NULL cl_int local_err; if (err == NULL) err = &local_err;
#define RETURN_ON_ERROR if (CL_SUCCESS != err) return err;

/**
 * @brief Get the GPU devices to run the computation on.
 * 
 * @return the OpenCL error code. Initializes the devices and command queues in globals if err = CL_SUCCESS.
 */
cl_int gpu::initializeGpuData(size_t device_no) {
    cl_int err;
    cl_uint numPlatforms;
    cl_uint numDevices;

    cl_platform_id *platforms = new cl_platform_id();
    cl_device_id *devices_local = new cl_device_id[MAX_DEVICES];

    // Get platforms.
    err = clGetPlatformIDs(1, platforms, &numPlatforms);
    if (CL_SUCCESS != err) {
        delete platforms;
        delete[] devices_local;
        return err;
    }

    // Get GPU devices from the platform.
    err = clGetDeviceIDs(*platforms, CL_DEVICE_TYPE_GPU, MAX_DEVICES, devices_local, &numDevices);
    if (CL_SUCCESS != err) {
        delete platforms;
        delete[] devices_local;
        return err;
    }

    devices = new ListWithSize<cl_device_id>();
    devices -> items = devices_local;
    devices -> num_items = numDevices;

    // Device selector
    if (device_no != -1 && device_no < devices -> num_items) {
        devices -> items[0] = devices -> items[device_no];
        devices -> num_items = 1;
    }

    delete platforms;

    context = new cl_context();
    *context = clCreateContext(NULL, devices -> num_items, devices -> items, NULL, NULL, &err);

    if (CL_SUCCESS != err) {
        FOR_ALL_DEVICES(clReleaseDevice(devices -> items[i]))

        delete[] devices_local;
        return err;
    }

    cl_command_queue *queues = new cl_command_queue[devices -> num_items];
    for (size_t i = 0; i < devices -> num_items; i++) {
        queues[i] = clCreateCommandQueueWithProperties(*context, devices -> items[i], NULL, &err);

        if (CL_SUCCESS != err) {
            clReleaseContext(*context);

            FOR_ALL_DEVICES(clReleaseDevice(devices -> items[i]))

            delete[] devices -> items;
            delete[] queues;

            return err;
        }

    }
    
    command_queues = new ListWithSize<cl_command_queue>();
    command_queues -> num_items = devices -> num_items;
    command_queues -> items = queues;

    available_kernels = new std::map<const char*, cl_kernel>();

    return CL_SUCCESS;
}

/**
 * @brief Cleans up all GPU data when done, freeing device and kernel global variables.
 */
void gpu::destroyGpuData() {
    for (auto i = available_kernels -> begin(); i != available_kernels -> end(); i++) {
        clReleaseKernel(i -> second);
    }

    FOR_ALL_DEVICES(clReleaseCommandQueue(command_queues -> items[i]));

    clReleaseContext(*context);

    FOR_ALL_DEVICES(clReleaseDevice(devices -> items[i]));

    delete[] devices -> items;
    delete[] command_queues -> items;

    delete devices;
    delete command_queues;
    delete context;
    delete available_kernels;
}

// This function generated using AI
GpuProps gpu::getDeviceProps(cl_device_id device) {
    GpuProps output = GpuProps();
    cl_int err;

    // get the device name
    char name[128];
    err = clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(name), &name, NULL);
    output.name = name;

    // get the device vendor
    char vendor[128];
    err = clGetDeviceInfo(device, CL_DEVICE_VENDOR, sizeof(vendor), &vendor, NULL);

    output.vendor = vendor;

    // get the maximum compute units
    err = clGetDeviceInfo(device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(output.max_compute_units), &output.max_compute_units, NULL);

    // get the maximum work group size
    err = clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(output.max_work_group_size), &output.max_work_group_size, NULL);

    // get the maximum work item sizes
    err = clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(output.max_work_item_sizes), &output.max_work_item_sizes, NULL);

    // get the total size of shared memory per block
    err = clGetDeviceInfo(device, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(cl_ulong), &output.shared_per_SM, NULL);
    output.err = err;

    return output;
}

ListWithSize<size_t> gpu::makeDim2(size_t fst, size_t snd) {
    ListWithSize<size_t> lst = ListWithSize<size_t>(2);
    lst.items[0] = fst;
    lst.items[1] = snd;

    return lst;
}

ListWithSize<size_t> gpu::makeDim3(size_t fst, size_t snd, size_t trd) {
    ListWithSize<size_t> lst = ListWithSize<size_t>(3);
    lst.items[0] = fst;
    lst.items[1] = snd;
    lst.items[2] = trd;

    return lst;
}

cl_kernel gpu::compileKernel(const char *kernelCode, const char *kernelFunctionName, cl_int *err) {
    SET_ERROR_IF_NULL;

    // Find and compile the program to be run on the GPU.
    cl_program program = clCreateProgramWithSource(*context, 1, &kernelCode, NULL, err);
    if (CL_SUCCESS != *err) return cl_kernel();

    clBuildProgram(program, devices -> num_items, devices -> items, NULL, NULL, NULL);

    cl_kernel kernel = clCreateKernel(program, kernelFunctionName, err);

    clReleaseProgram(program);

    return kernel;
}

cl_kernel gpu::compileKernelIfNotExists(const char *kernel, const char *kernelName, cl_int *err) {
    SET_ERROR_IF_NULL;
    
    if (available_kernels -> find(kernelName) == available_kernels -> end()) {
        (*available_kernels)[kernelName] = compileKernel(kernel, kernelName, err);
    }

    return (*available_kernels)[kernelName];
}

bool gpu::kernelExists(const char *kernelName) {
    return available_kernels -> find(kernelName) != available_kernels -> end();
}

cl_kernel gpu::compileKernelFromFile(const char *filename, const char *kernelName, cl_int *err) {
    SET_ERROR_IF_NULL;
    
    if (!kernelExists(kernelName)) {
        const char *kernel_string = io::readKernel(filename);
        compileKernelIfNotExists(kernel_string, kernelName, err);
    }

    return (*available_kernels)[kernelName];
}

cl_int gpu::setKernelArgs(cl_kernel &kernel, const ListWithSize<cl_mem*> &args) {
    cl_int err = CL_SUCCESS;

    for (int i = 0; i < args.num_items; i++) {
        clSetKernelArg(kernel, i, sizeof(args.items[i]), (void*) args.items[i]);
    }

    return err;
}

cl_mem gpu::deviceOutputAllocate(size_t nbytes, cl_int *err) {
    SET_ERROR_IF_NULL;
    return clCreateBuffer(*context, CL_MEM_WRITE_ONLY, nbytes, NULL, err);
}

cl_mem gpu::deviceIntermediateAllocate(size_t nbytes, cl_int *err) {
    SET_ERROR_IF_NULL;
    return clCreateBuffer(*context, CL_MEM_READ_WRITE, nbytes, NULL, err);
}

cl_int gpu::launchKernel(cl_kernel kernel, ListWithSize<size_t> gridDim, ListWithSize<size_t> blockDim) {
    cl_int err;

    FOR_ALL_DEVICES(
        err = clEnqueueNDRangeKernel(command_queues -> items[i], kernel, gridDim.num_items, NULL, gridDim.items, blockDim.items, 0, NULL, NULL);
        RETURN_ON_ERROR;
    )

    FOR_ALL_DEVICES(
        err = clFinish(command_queues -> items[i]); 
        RETURN_ON_ERROR;
    )

    return CL_SUCCESS;
}