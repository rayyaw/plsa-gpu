#pragma once

#include <CL/cl.h>
#include <map>
#include <string>

#include "../utils/listWithSize.h"

using std::map;
using std::string;

#define MAX_DEVICES 16
#define FOR_ALL_DEVICES(stmt) for (size_t i = 0; i < devices -> num_items; i++) {stmt;}
#define SET_ERROR_IF_NULL cl_int local_err; if (err == NULL) err = &local_err;
#define RETURN_ON_ERROR if (CL_SUCCESS != err) return err;

extern utils::ListWithSize<cl_device_id> *devices;
extern utils::ListWithSize<cl_command_queue> *command_queues;
extern cl_context *context;
extern map<const char*, cl_kernel> *available_kernels;

namespace gpu {
    // Should be called once, at the start of the program. Returns the error value.
    // Optionally, you can set device_no to specify a specific GPU device.
    cl_int initializeGpuData(size_t device_no=-1);

    // Should be called once, at the end of the program
    void destroyGpuData();

    struct GpuProps {
        // General properties
        string name;
        string vendor;

        cl_uint max_compute_units; // number of SMs
        size_t max_work_group_size; // max block size
        size_t max_work_item_sizes[3]; // how many blocks can we have

        // Per-SM properties
        cl_ulong shared_per_SM;

        cl_int err;
    };

    GpuProps getDeviceProps(cl_device_id device);

    // Saves the hassle of creating these manually
    // Create a ListWithSize of dimension 1, 2, or 3 to store the grid or block dimensions
    utils::ListWithSize<size_t> makeDim1(size_t fst);
    utils::ListWithSize<size_t> makeDim2(size_t fst, size_t snd);
    utils::ListWithSize<size_t> makeDim3(size_t fst, size_t snd, size_t trd);

    /**
     * @brief Create the GPU kernel by compiling the given code. Must be destroyed manually. May be used multiple times.
     * You should not call this function, instead you should use compileKernelIfNotExists().
     * 
     * @param kernelCode The kernel code (as string)
     * @param kernelFunctionName The name of the kernel function to run
     * @return cl_kernel The generated kernel. Modifies err on failure.
     */
    cl_kernel compileKernel(const char *kernelCode, const char *kernelFunctionName, cl_int *err);

    /**
     * @brief Compile an OpenCL kernel (if not already done), then return it.
     * 
     * @param kernel The kernel source code as a string
     * @param kernelName The function name within the kernel
     * @param err Modified on failure. If this is not equal to CL_SUCCESS, an error has occurred.
     * @return cl_kernel The kernel with the provided source. Will be released automatically on gpu::destroyGpuData().
     */
    cl_kernel compileKernelIfNotExists(const char *kernel, const char *kernelName, cl_int *err);

    /**
     * @brief Check if the specified kernel is already compiled
     * 
     * @param kernelName The kernel name
     * @return true if the kernel is already compiled
     */
    bool kernelExists(const char *kernelName);

    /**
     * @brief Compile the kernel from the file, if it doesn't exist already.
     * 
     * @param filename The name of the file containing the kernel
     * @param kernelName The name of the kernel
     * @param err Modified on failure. If this is not equal to CL_SUCCESS, an error has occurred.
     * @return The kernel with the provided source. Will be released automatically on gpu::destroyGpuData().
     */
    cl_kernel compileKernelFromFile(const char *filename, const char *kernelName, cl_int *err);

    /**
     * @brief Create a device memory array, and copy the host memory onto it
     * 
     * @param hostMem The host memory, along with its length. num_items should be the number of ITEMS, not bytes.
     * @param err Modified on failure. If this is not equal to CL_SUCCESS, an error has occurred.
     * @return cl_mem The device global memory that was allocated. It is your responsibility to release it.
     */
    template <typename T>
    cl_mem hostToDeviceCopy(utils::ListWithSize<T> hostMem, cl_int *err) {
        SET_ERROR_IF_NULL;
        return clCreateBuffer(*context, CL_MEM_COPY_HOST_PTR, hostMem.num_items * sizeof(T), hostMem.items, err);
    }

    /**
     * @brief Create a device memory array, and copy the host memory onto it
     * 
     * @param hostMem The host memory, along with its length. num_items should be the number of ITEMS, not bytes.
     * @param nitems The number of items to use.
     * @param err Modified on failure. If this is not equal to CL_SUCCESS, an error has occurred.
     * @return cl_mem The device global memory that was allocated. It is your responsibility to release it.
     */
    template <typename T>
    cl_mem hostToDeviceCopy(T *hostMem, size_t nitems, cl_int *err) {
        SET_ERROR_IF_NULL;
        return clCreateBuffer(*context, CL_MEM_COPY_HOST_PTR, nitems * sizeof(T), hostMem, err);
    }

    /**
     * @brief Create a device memory array that is write-only (for output)
     * 
     * @param nbytes The number of bytes to copy
     * @param err Modified on failure. If this is not equal to CL_SUCCESS, an error has occurred.
     * @return cl_mem The device global memory that was allocated. It is your responsibility to release it.
     */
    cl_mem deviceOutputAllocate(size_t nbytes, cl_int *err);

    /**
     * @brief Create a device memory array that is read-write (for intermediate values)
     * 
     * @param nbytes The number of bytes to use
     * @param err Modified on failure. If this is not equal to CL_SUCCESS, an error has occurred.
     * @return cl_mem The device global memory that was allocated. It is your responsibility to release it.
     */
    cl_mem deviceIntermediateAllocate(size_t nbytes, cl_int *err);

    /**
     * @brief Copy device memory back to the host. Blocking call.
     * 
     * @param deviceMem The device memory to copy
     * @param hostMem The host buffer to copy into. Must be allocated before calling the function. num_items should already be set
     * @return cl_int An error code, or CL_SUCCESS if no error occurred
     */
    template <typename T>
    cl_int copyDeviceToHost(cl_mem deviceMem, utils::ListWithSize<T> hostMem) {
        return clEnqueueReadBuffer(command_queues -> items[0], deviceMem, CL_TRUE, 0, hostMem.num_items * sizeof(T), hostMem.items, 0, NULL, NULL);
    }

    /**
     * @brief Copy device memory back to the host. Blocking call.
     * 
     * @param deviceMem The device memory to copy
     * @param hostMem The host buffer to copy into. Must be allocated before calling the function. num_items should already be set
     * @return cl_int An error code, or CL_SUCCESS if no error occurred
     */
    template <typename T>
    cl_int copyDeviceToHost(cl_mem deviceMem, T *hostMem, size_t nitems) {
        return clEnqueueReadBuffer(command_queues -> items[0], deviceMem, CL_TRUE, 0, nitems * sizeof(T), hostMem, 0, NULL, NULL);
    }


    /**
     * @brief Launch the specified kernel, and block until it's done.
     * 
     * @param kernel The kernel to launch
     * @param gridDim Grid dimensions. Should be the number of THREADS, not blocks
     * @param blockDim Block dimensions
     * @return cl_int An error code, or CL_SUCCESS if no error occurred
     */
    cl_int launchKernel(cl_kernel kernel, utils::ListWithSize<size_t> gridDim, utils::ListWithSize<size_t> blockDim);
}