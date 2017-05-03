#include <stdio.h>
#include "lib_opencl.h"
#include "lib_signal.h"

int main(int argc, const char * argv[])
{
    
    cl_device_id my_device_list[10];
    cl_uint      num_dev;
    cl_int       err;
    
    /* Get and Print device information.
     */
    {
        clGetDeviceIDs(NULL, CL_DEVICE_TYPE_ALL, 10, my_device_list, &num_dev);
        clPrintAllAvaliableDevicesInfo(my_device_list, num_dev);
    }
    
    /* Initialize signal analysis component.
     */
    {
        signalInit(my_device_list, num_dev, &err);
    }

    /* Test 1D DCT
     */
    {
        const int matrix_size  = 10;
        float  input_matrix[matrix_size] = {0.218418, 0.956318, 0.829509, 0.561695,
            0.415307, 0.066119, 0.257578, 0.109957,
            0.043829, 0.633966};
        float  output_matrix[matrix_size];
        float  inverse_matrix[matrix_size];
        signal_matrix_t signal_input;
        signal_matrix_t signal_dct;
        signal_matrix_t signal_idct;
        
        /* Initialize 1D matrix for computation.
         */
        for (int i = 0; i < matrix_size; i += 1)
        {
            output_matrix[i]  = 0;
            inverse_matrix[i] = 0;
        }
        
        signal_input.input_dims[0] = matrix_size;
        signal_input.input_dims[1] = 0;
        signal_input.signal        = input_matrix;
        
        signal_dct.signal = output_matrix;
        signal_dct.input_dims[0] = matrix_size;
        signal_dct.input_dims[1] = 0;
        
        signal_idct.signal = inverse_matrix;
        signal_idct.input_dims[0] = matrix_size;
        signal_idct.input_dims[1] = 0;
        
        
        signalCompute(SIGNAL_1D_DCT,
                      &signal_input,
                      &signal_dct,
                      &err);
        
        signalCompute(SIGNAL_1D_IDCT,
                      &signal_dct,
                      &signal_idct,
                      &err);
        
        if (err != CL_SUCCESS)
        {
            printf("Signal Error: %d.\n", err);
            return 1;
        }
        else
        {
            /* Print resutls.
             */
            printf("\nPrinting Results:\n\tInput Matrix:\n");
            for (int i = 0; i < matrix_size; i += 1)
            {
                printf("\t%f%s", input_matrix[i], (i != (matrix_size -1)) ? "," : "\n");
            }
            printf("\n\tOutput Matrix:\n");
            for (int i = 0; i < matrix_size; i += 1)
            {
                printf("\t%f%s", output_matrix[i], (i != (matrix_size -1)) ? "," : "\n");
            }
            printf("\n\tInverse Matrix:\n");
            for (int i = 0; i < matrix_size; i += 1)
            {
                printf("\t%f%s", inverse_matrix[i], (i != (matrix_size -1)) ? "," : "\n");
            }
            
        }
    }
    
    /* Test 2D DCT
     */
    {
        const int matrix_size = 8;
        float input_matrix[matrix_size][matrix_size];
        float output_matrix[matrix_size][matrix_size];
        float inverse_matrix[matrix_size][matrix_size];
        
        signal_matrix_t input_signal;
        signal_matrix_t signal_dct;
        signal_matrix_t signal_idct;
        
        /* Initialize input_matrix with know values and output_matrix and inverse_matrix
         * with zeros.
         */
        for (int i = 0; i < matrix_size; i += 1)
        {
            for (int j = 0 ; j < matrix_size; j += 1)
            {
                input_matrix[i][j]   = 10*(i + j);
                output_matrix[i][j]  = 0;
                inverse_matrix[i][j] = 0;
            }
        }
        
        input_signal.input_dims[0] = matrix_size;
        input_signal.input_dims[1] = matrix_size;
        input_signal.signal        = (float *)input_matrix;
        
        signal_dct.input_dims[0] = matrix_size;
        signal_dct.input_dims[1] = matrix_size;
        signal_dct.signal        = (float *)output_matrix;
        
        signal_idct.input_dims[0] = matrix_size;
        signal_idct.input_dims[1] = matrix_size;
        signal_idct.signal        = (float *)inverse_matrix;
        
        signalCompute(SIGNAL_2D_DCT,
                      &input_signal,
                      &signal_dct,
                      &err);
        
        signalCompute(SIGNAL_2D_IDCT,
                      &signal_dct,
                      &signal_idct,
                      &err);
        
        
        /* Print results:
         */
        if (err != CL_SUCCESS)
        {
            printf("Signal Error: %d", err);
            return 1;
        }
        else
        {
            printf("\nPrinting results:\n\tInput_Matrix:\n");
            /* Print input_matrix values.
             */
            for (int i = 0; i < matrix_size; i += 1)
            {
                for (int j = 0; j < matrix_size; j += 1)
                {
                    printf("\t%f%s", input_matrix[i][j], (j != (matrix_size-1) ? "," : "\n"));
                }
            }
            
            printf("\n\tOutput_Matrix:\n");
            /* Print output_matrix values.
             */
            for (int i = 0; i < matrix_size; i += 1)
            {
                for (int j = 0; j < matrix_size; j += 1)
                {
                    printf("\t%f%s", output_matrix[i][j], (j != (matrix_size-1) ? "," : "\n"));
                }
            }
            
            printf("\n\tInverse_Matrix:\n");
            /* Print inverse_matrix values.
             */
            for (int i = 0; i < matrix_size; i += 1)
            {
                for (int j = 0; j < matrix_size; j += 1)
                {
                    printf("\t%f%s", inverse_matrix[i][j], (j != (matrix_size-1) ? "," : "\n"));
                }
            }
        }
    }

    return 0;
}
