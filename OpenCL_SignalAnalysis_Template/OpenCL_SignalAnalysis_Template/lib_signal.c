
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>

#include "lib_opencl.h"
#include "lib_signal.h"

//////////////////////////////////////////////////////////////////////////////////////////////////

#define ERR_DEVICE_CONTEXT_CREATION_NOK 0
#define ERR_KERNEL_OBJS_CREATION_NOK    1
#define ERR_SIGNAL_OPERATION_NOK        2
#define ERR_BUFFER_CREATION_NOK         3
#define ERR_WRITE_BUFFER_NOK            4
#define ERR_SETTING_ARGUMENTS_NOK       5
#define ERR_READ_BUFFER_NOK             6

#define INFO_DEVICE_CONTEXT_CREATION_OK (ERR_DEVICE_CONTEXT_CREATION_NOK)
#define INFO_KERNEL_OBJS_CREATION_NOK   (ERR_KERNEL_OBJS_CREATION_NOK)
//////////////////////////////////////////////////////////////////////////////////////////////////

static volatile cl_device_id * dev_list = NULL;
static cl_int     dev_cnt = 0;

static cl_context       signal_context;
static cl_command_queue signal_cmd_queue;
static cl_kernel        signal_kernel_list[KERNEL_PRG_CNT];


static char * kernel_name_list[KERNEL_PRG_CNT] = SIGNAL_KERNEL_LIST_NAMES;

//////////////////////////////////////////////////////////////////////////////////////////////////
static void printSignalErrorMsg(int err_id);
static void printSignalInfoMsg(int msg_id);
//////////////////////////////////////////////////////////////////////////////////////////////////


static void printSignalErrorMsg(int err_id)
{
    switch (err_id)
    {
        case ERR_DEVICE_CONTEXT_CREATION_NOK:
        {
            printf("Error Signal analysis component: Created device and context ... NOK.\n");
            break;
        }
        case ERR_KERNEL_OBJS_CREATION_NOK:
        {
            printf("Error Signal analysis component: Create kernel objects ... NOK.\n");
            break;
        }
        case ERR_BUFFER_CREATION_NOK:
        {
            printf("Error Signal analysis component: Create memory buffer for internal buffer ... NOK.\n");
            break;
        }
        case ERR_WRITE_BUFFER_NOK:
        {
            printf("Error Signal analysis component: Writing to buffer ... NOK.\n");
            break;
        }
        case ERR_SETTING_ARGUMENTS_NOK:
        {
            printf("Error Signal analysis component: Setting arguments for kernel ... NOK.\n");
            break;
        }
        case ERR_READ_BUFFER_NOK:
        {
            printf("Error Signal analysis component: Reading from buffer ... NOK.\n");
            break;
        }
        default:
            break;
    }
}

static void printSignalInfoMsg(int msg_id)
{
    switch (msg_id)
    {
        case INFO_DEVICE_CONTEXT_CREATION_OK:
        {
            printf("Info Signal analysis component: Created device and context ... OK.\n");
            break;
        }
        case INFO_KERNEL_OBJS_CREATION_NOK:
        {
            printf("Info Signal analysis component: Create kernel objects ... OK\n");
            break;
        }
        default:
            break;
    }
}
//////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////

void signalInit(const cl_device_id * const device_list,
                cl_int               num_dev,
                cl_int       * const ret_err)
{
    
    /* Initialize dev_list value and device count.
     */
    dev_list = (volatile cl_device_id *)device_list;
    dev_cnt  = num_dev;
    
    /* Get device list and if there is no avaliable device list then
     * return CPU device and print a warning.
     * Then create Context and Command queue for selected devices.
     */
    clCreateDeviceAndContext((cl_device_id * const )dev_list,
                             dev_cnt,
                             &signal_context,
                             &signal_cmd_queue,
                             ret_err);
    
    if (*ret_err != CL_SUCCESS)
    {
        printSignalErrorMsg(ERR_DEVICE_CONTEXT_CREATION_NOK);
        return;
    }
    else
    {
        printSignalInfoMsg(INFO_DEVICE_CONTEXT_CREATION_OK);
    }
    
    /* Create program and kernel objects.
     */
    clCreateKernelObjsForContext(&signal_context,
                                 (SIGNAL_KERNEL_FILE_NAME),
                                 (const char **)kernel_name_list,
                                 (KERNEL_PRG_CNT),
                                 signal_kernel_list,
                                 ret_err);
    if (*ret_err != CL_SUCCESS)
    {
        printSignalErrorMsg(ERR_KERNEL_OBJS_CREATION_NOK);
        return;
    }
    
    *ret_err = CL_SUCCESS;
}

void signalCompute(int signal_operation,
                   signal_matrix_t * const input_signal,
                   signal_matrix_t * const ret_signal,
                   int             * const ret_err)
{
    /* Check the type of operation, based on operation type the following parameters
     * shall be defined:
     * 1- number of required buffers.
     * 2- set work group global size.
     * 3- set problem dimension value.
     * 4- set buffer size.
     * 5- set number of input buffers to be placed in kernel memory.
     * 6- set number of arguments.
     * 7- set start output buffer index.
     */
    
    cl_int num_buffer;
    cl_mem * kernel_buffer;
    size_t   global[2];
    size_t   buffer_size;
    size_t   num_input_buffer_write;
    size_t   num_arguments;
    size_t   start_output_buffer_index;
    cl_int   problem_dim;
    
    
    switch (signal_operation)
    {
            /*! For Signal 1D DCT/IDCT do the following:
             */
        case SIGNAL_1D_DCT:
        case SIGNAL_1D_IDCT:
        {
            /* Kernel API: __kernel void computeDCT1D(__global float * input_mat,
             *                                        __global float * ret_mat,
             *                                                 int     input_dim)
             */
            /*! \tSet number of buffers to two.
             */
            num_buffer = 2;
            /*! \tAllocate two elements in kernel buffer list.
             */
            kernel_buffer = (cl_mem *)malloc(num_buffer*sizeof(cl_mem));
            /*! \tSet work group size, global[0] = input dimension and global[1] = 0.
             */
            input_signal->input_dims[1] = (input_signal->input_dims[1] == 0) ? 1 : input_signal->input_dims[1];
            global[0] = input_signal->input_dims[0];
            global[1] = 0;
            /*! \tSet problem dimension to 1.
             */
            problem_dim = 1;
            /*! \tSet buffer Maximum size.
             */
            buffer_size = input_signal->input_dims[1] * input_signal->input_dims[0];
            /*! \tSet number of buffers to be written to 1.
             */
            num_input_buffer_write = 1;
            /*! \Set number of arguments to 3.
             */
            num_arguments = 3;
            /*! \Set start index for output buffer index to 1.
             */
            start_output_buffer_index = 1;
            /*! \Set ret_signal dimsions.
             */
            ret_signal->input_dims[0] = input_signal->input_dims[0];
            ret_signal->input_dims[1] = input_signal->input_dims[1];
            
            break;
        }
        case SIGNAL_2D_DCT:
        case SIGNAL_2D_IDCT:
        {
            /* Kernel API: __kernel void computeDCT2D(__global float * input_mat,
             *                                        __global float * ret_mat,
             *                                                 int     input_mat_dim_x,
             *                                                 int     input_mat_dim_y)
             */
            
            /*! \tSet number of buffers to two.
             */
            num_buffer = 2;
            /*! \tAllocate two elements in kernel buffer list.
             */
            kernel_buffer = (cl_mem *)malloc(num_buffer*sizeof(cl_mem));
            /*! \tSet work group size, global[0,1] = input dimension.
             */
            input_signal->input_dims[1] = (input_signal->input_dims[1] == 0) ? 1 : input_signal->input_dims[1];
            global[0] = input_signal->input_dims[0];
            global[1] = input_signal->input_dims[1];
            /*! \tSet problem dimension to 2.
             */
            problem_dim = 2;
            /*! \tSet buffer Maximum size.
             */
            buffer_size = input_signal->input_dims[1] * input_signal->input_dims[0];
            /*! \tSet number of buffers to be written to 1.
             */
            num_input_buffer_write = 1;
            /*! \Set number of arguments to 4.
             */
            num_arguments = 4;
            /*! \Set start index for output buffer index to 1.
             */
            start_output_buffer_index = 1;
            /*! \Set ret_signal dimsions.
             */
            ret_signal->input_dims[0] = input_signal->input_dims[0];
            ret_signal->input_dims[1] = input_signal->input_dims[1];
            
            break;
        }
        default :
        {
            printSignalErrorMsg(ERR_SIGNAL_OPERATION_NOK);
            *ret_err = !(CL_SUCCESS);
            return;
        }
    }
    
    /*! Create buffer for kernel.
     */
    for (size_t i = 0; i < num_buffer; i += 1)
    {
        kernel_buffer[i] = clCreateBuffer(signal_context,
                                          CL_MEM_READ_WRITE,
                                          (buffer_size * sizeof(float)),
                                          NULL,
                                          ret_err);
        if (*ret_err != CL_SUCCESS)
        {
            free(kernel_buffer);
            printSignalErrorMsg(ERR_BUFFER_CREATION_NOK);
            return;
        }
    }
    
    /*! Write input buffers.
     */
    for (size_t i = 0; i < num_input_buffer_write; i += 1)
    {
        *ret_err = clEnqueueWriteBuffer(signal_cmd_queue,
                                        kernel_buffer[i],
                                        CL_TRUE,
                                        0,
                                        (buffer_size * sizeof(float)),
                                        (const void *)input_signal->signal,
                                        0,
                                        NULL,
                                        NULL);
        if (*ret_err != CL_SUCCESS)
        {
            free(kernel_buffer);
            printSignalErrorMsg(ERR_WRITE_BUFFER_NOK);
            return;
        }
    }
    
    /*! Wait until copy is complete.
     */
    clFinish(signal_cmd_queue);
    
    /*! Set kernel arguments.
     */
    *ret_err  = 0;
    
    for (size_t i = 0; i < num_arguments; i += 1)
    {
        if (i < num_buffer)
        {
            /* Set buffers arguments.
             */
            *ret_err |= clSetKernelArg(signal_kernel_list[signal_operation],
                                       (cl_int)i,
                                       (sizeof(cl_mem)),
                                       &kernel_buffer[i]);
        }
        else if(i < (problem_dim + num_buffer))
        {
            /* Set input matrix dimensions.
             */
            *ret_err |= clSetKernelArg(signal_kernel_list[signal_operation],
                                       (cl_int)i,
                                       (sizeof(int)),
                                       &input_signal->input_dims[(i - num_buffer)]);
        }
        else
        {
            /* Left for future usage.
             */
        }
        
        if (*ret_err != CL_SUCCESS)
        {
            free(kernel_buffer);
            printSignalErrorMsg(ERR_SETTING_ARGUMENTS_NOK);
            return;
        }
    }
    
    /*! Enqueue data task execution.
     */
    *ret_err = clEnqueueNDRangeKernel(signal_cmd_queue,
                                      signal_kernel_list[signal_operation],
                                      problem_dim,
                                      NULL,
                                      global,
                                      NULL,
                                      0,
                                      NULL,
                                      NULL);
    /*! Wait for command queue to finish.
     */
    clFinish(signal_cmd_queue);
    
    /*! Read kernel output buffers.
     */
    for (size_t i = start_output_buffer_index; i < num_buffer; i += 1)
    {
        *ret_err = clEnqueueReadBuffer(signal_cmd_queue,
                                       kernel_buffer[i],
                                       CL_TRUE,
                                       0,
                                       (buffer_size * sizeof(float)),
                                       (void *)ret_signal->signal,
                                       0,
                                       NULL,
                                       NULL);
        if (*ret_err != CL_SUCCESS)
        {
            printSignalErrorMsg(ERR_READ_BUFFER_NOK);
            return;
        }
        
        /*! Wait for read to be complete.
         */
        clFinish(signal_cmd_queue);
        
    }
    
    /*! Clean buffers.
     */
    for (int i = 0; i < num_buffer; i += 1)
    {
        clReleaseMemObject(kernel_buffer[i]);
    }
    
    *ret_err = CL_SUCCESS;
    
}
//////////////////////////////////////////////////////////////////////////////////////////////////
