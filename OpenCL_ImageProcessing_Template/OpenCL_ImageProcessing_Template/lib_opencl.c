
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <string.h>

#include "lib_opencl.h"

#define ERR_INVALID_SOURCE_CODE    -1
#define ERR_SRC_CODE_NOK           -2
#define ERR_SRC_BUILD_FAILED       -3
#define ERR_CREATE_KERNEL_NOK      -4
#define ERR_GET_DEVICE_INFO_NOK    -5
#define ERR_INVALID_CREATE_CONTEXT -6
#define ERR_INVALID_CREATE_COMMAND -7

#define INFO_VALID_SOURCE_CODE    (ERR_INVALID_SOURCE_CODE)
#define INFO_CREATE_KERNEL_OK     (ERR_CREATE_KERNEL_NOK)
#define INFO_GET_DEVICE_INFO_OK   (ERR_GET_DEVICE_INFO_NOK)
#define INFO_VALID_CREATE_CONTEXT (ERR_INVALID_CREATE_CONTEXT)
#define INFO_VALID_CREATE_COMMAND (ERR_INVALID_CREATE_COMMAND)

//////////////////////////////////////////////////////////////////////////////////////////////////

static char * LoadProgramSrc(const char * filename);
static void printOpenCLErrorMsg(int err);
static void printOpenCLInfoMsg(int msg);

//////////////////////////////////////////////////////////////////////////////////////////////////

static char * LoadProgramSrc(const char * filename)
{
    struct stat stat_buffer;
    FILE        *file_ptr;
    char        *ret_src;
    
    /*! Open file.
     */
    file_ptr = fopen(filename, "r");
    
    if (file_ptr == 0)
    {
        ret_src = 0;
    }
    else
    {
        /*! Get file information.
         */
        stat(filename, &stat_buffer);
        
        /*! Allocate memory for return string.
         */
        ret_src = (char *)malloc(stat_buffer.st_size + 1);
        
        /*! Read file and update ret_src.
         */
        fread(ret_src, stat_buffer.st_size, 1, file_ptr);
        
        /*! Add termination string.
         */
        ret_src[stat_buffer.st_size] = '\0';
    }
    
    return (ret_src);
}

static void printOpenCLErrorMsg(int err)
{
    switch (err)
    {
        case ERR_INVALID_SOURCE_CODE:
        {
            printf("Error OpenCL: Load source program ... NOK.\n");
            break;
        }
        case ERR_SRC_BUILD_FAILED:
        {
            printf("Error OpenCL: Source code build ... NOK.\n");
            break;
        }
        case ERR_CREATE_KERNEL_NOK:
        {
            printf("Error OpenCL: Kernel object creation ... NOK.\n");
            break;
        }
        case ERR_GET_DEVICE_INFO_NOK:
        {
            printf("Error OpenCL: Device prope ... NOK.\n");
            break;
        }
        case ERR_INVALID_CREATE_CONTEXT:
        {
            printf("Error OpenCL: Create device context ... NOK.\n");
            break;
        }
        case ERR_INVALID_CREATE_COMMAND:
        {
            printf("Error OpenCL: Create command queue ... NOK.\n");
            break;
        }
        default:
        {
            break;
        }
            
    }
}

static void printOpenCLInfoMsg(int msg)
{
    switch (msg)
    {
        case INFO_VALID_SOURCE_CODE:
        {
            printf("Info OpenCL: Load source program ... OK.\n");
            break;
        }
        case INFO_CREATE_KERNEL_OK:
        {
            printf("Info OpenCL: Kernel Code Compilation ... OK.\n");
        }
        case INFO_GET_DEVICE_INFO_OK:
        {
            printf("Info OpenCL: All devices prope ... OK.\n");
            break;
        }
        case INFO_VALID_CREATE_CONTEXT:
        {
            printf("Info OpenCL: Create device context ... OK.\n");
            break;
        }
        case INFO_VALID_CREATE_COMMAND:
        {
            printf("Info OpenCL: Create command queue ... OK.\n");
            break;
        }
        default:
        {
            break;
        }
    }
}
//////////////////////////////////////////////////////////////////////////////////////////////////


//////////////////////////////////////////////////////////////////////////////////////////////////

void clPrintAllAvaliableDevicesInfo(cl_device_id * usr_device_list,
                                    cl_uint num_devices)
{
    cl_device_type device_type;
    
    cl_int         i;
    cl_bool        ret_bool;
    char ret_string[1024];
    size_t ret_count;
    
    printf("Info: Number of devices: %d.\n", num_devices);
    
    for (i = 0;  i < num_devices; i += 1)
    {
        clGetDeviceInfo(usr_device_list[i],
                        CL_DEVICE_NAME,
                        sizeof(ret_string),
                        ret_string,
                        NULL);
        printf("Device Name: %s.\n",ret_string);
        
        clGetDeviceInfo(usr_device_list[i],
                        CL_DEVICE_VENDOR,
                        sizeof(ret_string),
                        ret_string,
                        NULL);
        printf("\tInfo: Device Vendor: %s.\n", ret_string);
        
        clGetDeviceInfo(usr_device_list[i],
                        CL_DEVICE_TYPE,
                        sizeof(device_type),
                        &device_type,
                        NULL);
        switch (device_type)
        {
            case CL_DEVICE_TYPE_CPU:         {strcpy(ret_string, "CPU");         break;}
            case CL_DEVICE_TYPE_GPU:         {strcpy(ret_string, "GPU");         break;}
            case CL_DEVICE_TYPE_ACCELERATOR: {strcpy(ret_string, "ACCELERATOR"); break;}
            case CL_DEVICE_TYPE_DEFAULT:     {strcpy(ret_string, "DEFAULT");     break;}
            default:                         {break;}
        }
        printf("\tInfo: Device Type: %s.\n", ret_string);
        
        ret_count = 0;
        clGetDeviceInfo(usr_device_list[i],
                        CL_DEVICE_MAX_COMPUTE_UNITS,
                        sizeof(ret_count),
                        &ret_count,
                        NULL);
        printf("\tInfo: Device Max Compute Units: %zu.\n", (ret_count));
        
        ret_count = 0;
        clGetDeviceInfo(usr_device_list[i],
                        CL_DEVICE_MAX_WORK_GROUP_SIZE,
                        sizeof(ret_count),
                        &ret_count,
                        NULL);
        printf("\tInfo: Device Max Work Group Size: %zu. \n", (ret_count));
        
        ret_count = 0;
        clGetDeviceInfo(usr_device_list[i],
                        CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS,
                        sizeof(ret_count),
                        &ret_count,
                        NULL);
        printf("\tInfo: Device Max Work Item Dimensions: %zu.\n", (ret_count));
        
        ret_count = 0;
        clGetDeviceInfo(usr_device_list[i],
                        CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE,
                        sizeof(ret_count),
                        &ret_count,
                        NULL);
        printf("\tInfo: Device Max Constant Buffer Size: %zu Kbytes.\n", (ret_count/1024));
        
        ret_count = 0;
        clGetDeviceInfo(usr_device_list[i],
                        CL_DEVICE_GLOBAL_MEM_SIZE,
                        sizeof(ret_count),
                        &ret_count,
                        NULL);
        printf("\tInfo: Device Global Memory Size: %zu Mbytes.\n",(ret_count/(2*1024)));
        
        clGetDeviceInfo(usr_device_list[i],
                        CL_DEVICE_IMAGE_SUPPORT,
                        sizeof(ret_bool),
                        &ret_bool,
                        NULL);
        
        printf("\tInfo: Device Support Image: %s.\n", (ret_bool == CL_TRUE) ? "YES" : "NO");
        
        ret_count = 0;
        clGetDeviceInfo(usr_device_list[i],
                        CL_DEVICE_MAX_SAMPLERS,
                        sizeof(ret_count),
                        &ret_count,
                        NULL);
        printf("\tInfo: Device Maximum Samplers: %zu.\n",ret_count);
    }
}

void clCreateKernelObjsForContext(const cl_context * const device_context,
                                  const char  *filename,
                                  const char  *prg_name[],
                                  cl_int      num_kernel,
                                  cl_kernel   * const ret_kernel,
                                  cl_int      * const ret_err)
{
    cl_int       err;
    cl_program   usr_prg;
    cl_uint      i;
    char         *src_code;
    
    /* Load source code.
     */
    src_code = LoadProgramSrc(filename);
    
    if (src_code == 0)
    {
        *ret_err = ERR_INVALID_SOURCE_CODE;
        printOpenCLErrorMsg(ERR_INVALID_SOURCE_CODE);
        return;
    }
    else
    {
        printOpenCLInfoMsg(INFO_VALID_SOURCE_CODE);
    }
    
    
    /* Create program with source object.
     */
    usr_prg = clCreateProgramWithSource((*device_context),
                                        1,
                                        (const char ** )&src_code,
                                        NULL,
                                        &err);
    
    if ((0 == usr_prg) || (err != CL_SUCCESS))
    {
        *ret_err = ERR_SRC_CODE_NOK;
        return;
    }
    
    /* Build program for all devices.
     */
    err = clBuildProgram(usr_prg,
                         0,
                         NULL,
                         NULL,
                         NULL,
                         NULL);
    if (err != CL_SUCCESS)
    {
        char   error_log[2048];
        size_t len;
        
        printOpenCLErrorMsg(ERR_SRC_BUILD_FAILED);
        
        /* Print build log in case of failure in compilation.
         */
        clGetProgramBuildInfo(usr_prg,
                              NULL,
                              CL_PROGRAM_BUILD_LOG,
                              sizeof(error_log),
                              error_log,
                              &len);
        printf("\tError log: %s", error_log);
        
        *ret_err = err;
        return;
    }
    
    for (i = 0; i < num_kernel; i += 1)
    {
        /* Create kernel objects for all functions found in user cl file.
         */
        ret_kernel[i] = clCreateKernel(usr_prg,
                                       prg_name[i],
                                       &err);
        if (err != CL_SUCCESS)
        {
            printOpenCLErrorMsg(ERR_CREATE_KERNEL_NOK);
            *ret_err = err;
            return;
        }
    }
    
    printOpenCLInfoMsg(INFO_CREATE_KERNEL_OK);
    
    /* Tear down usr_prg.
     */
    clReleaseProgram(usr_prg);
    free(src_code);
    
    *ret_err = CL_SUCCESS;
}

void clCreateDeviceAndContext(cl_device_id     * const device_list,
                              cl_int                   device_num,
                              cl_context       * const device_context,
                              cl_command_queue * const device_cmd_queue,
                              cl_int           * const ret_err)
{
    cl_int err;
    cl_int i;
    
    err = 0;
    
    for (i = 0; i < device_num; i += 1)
    {
        /* Get device information for index i.
         */
        err |= clGetDeviceIDs(NULL,
                              CL_DEVICE_TYPE_ALL,
                              1,
                              &device_list[i],
                              NULL);
    }
    
    if (err != CL_SUCCESS)
    {
        *ret_err = err;
        printOpenCLErrorMsg(ERR_GET_DEVICE_INFO_NOK);
        return;
    }
    else
    {
        printOpenCLInfoMsg(INFO_GET_DEVICE_INFO_OK);
    }
    
    /* Create context for GPU.
     * ToDo: Update the method for creating context.
     */
    *device_context = clCreateContext(0,
                                      1,
                                      &device_list[1],
                                      NULL,
                                      NULL,
                                      &err);
    
    /* Create command for GPU.
     * ToDo: Update the method for creating context.
     */
    if (err != CL_SUCCESS)
    {
        *ret_err = err;
        printOpenCLErrorMsg(ERR_INVALID_CREATE_CONTEXT);
        return;
    }
    else
    {
        printOpenCLInfoMsg(INFO_VALID_CREATE_CONTEXT);
    }
    
    *device_cmd_queue = clCreateCommandQueue(*device_context,
                                             device_list[1],
                                             0,
                                             &err);
    if (err != CL_SUCCESS)
    {
        *ret_err = err;
        printOpenCLErrorMsg(ERR_INVALID_CREATE_COMMAND);
        return;
    }
    else
    {
        printOpenCLInfoMsg(INFO_VALID_CREATE_COMMAND);
    }
    
    *ret_err = CL_SUCCESS;
}

void clCleanEnvironment(cl_context       * device_context,
                        cl_command_queue * device_cmd_queue,
                        cl_kernel        * kernel_list,
                        cl_int             num_kernel_list)
{
    cl_int i;
    
    for (i = 0;  i < num_kernel_list; i += 1)
    {
        clReleaseKernel(kernel_list[i]);
    }
    
    clReleaseContext(*device_context);
    clReleaseCommandQueue(*device_cmd_queue);
}
//////////////////////////////////////////////////////////////////////////////////////////////////
