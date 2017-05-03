#ifndef _LIB_SIGNAL_H_
#define _LIB_SIGNAL_H_

#include <OpenCL/OpenCL.h>
#include "lib_signal_cfg.h"

typedef struct
{
  float * signal;
  int     input_dims[2];
}signal_matrix_t;

extern void signalInit(const cl_device_id * const device_list,
                       cl_int               num_dev,
                       cl_int       * const ret_err);

extern void signalCompute(int signal_operation,
                          signal_matrix_t * const input_signal,
                          signal_matrix_t * const ret_signal,
                          int             * const ret_err);

#endif /* _LIB_SIGNAL_H_ */
