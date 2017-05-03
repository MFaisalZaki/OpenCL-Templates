#ifndef _LIB_OPENCL_H_
#define _LIB_OPENCL_H_

#include <OpenCL/OpenCL.h>


extern void clCreateKernelObjsForContext( const cl_context * const device_context,
                                         const char  *filename,
                                         const char  *prg_name[],
                                         cl_int      num_kernel,
                                         cl_kernel   * const ret_kernel,
                                         cl_int      * const ret_err);

extern void clCreateDeviceAndContext(cl_device_id     * const device_list,
                                     cl_int                   device_num,
                                     cl_context       * const device_context,
                                     cl_command_queue * const device_cmd_queue,
                                     cl_int           * const ret_err);

extern void clPrintAllAvaliableDevicesInfo(cl_device_id * usr_device_list,
                                           cl_uint num_devices);

extern void clCleanEnvironment(cl_context       * device_context,
                               cl_command_queue * device_cmd_queue,
                               cl_kernel        * kernel_list,
                               cl_int             num_kernel_list);

#endif /* _LIB_OPENCL_H_ */
