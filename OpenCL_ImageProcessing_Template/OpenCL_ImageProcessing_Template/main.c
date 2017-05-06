#include <stdio.h>
#include <stdlib.h>
#include "lib_opencl.h"
#include "lib_image.h"

#define IMAGE_INPUT_FILENAME "/Users/marwanfaisal/Desktop/OpenCL-Templates/OpenCL_ImageProcessing_Template/OpenCL_ImageProcessing_Template/test.ppm"

#define IMAGE_OUTPUT_FILENAME "/Users/marwanfaisal/Desktop/OpenCL-Templates/OpenCL_ImageProcessing_Template/OpenCL_ImageProcessing_Template/test_filter.ppm"

int main(int argc, const char *argv[])
{
    cl_device_id my_dev_list[10];
    cl_uint      num_dev;
    cl_int       err;
    
    opencl_image_t *input_opencl_image;
    opencl_image_t *filtered_opencl_image;
    
    ppm_image_t *read_image; // for now.
    
    /* Get and Print device information.
     */
    {
        clGetDeviceIDs(NULL, CL_DEVICE_TYPE_ALL, 10, my_dev_list, &num_dev);
        clPrintAllAvaliableDevicesInfo(my_dev_list, num_dev);
    }

    /* Initialize Image Component.
     */
    {
        imageInit(my_dev_list, num_dev, &err);
    }
    
    /* Read input image.                */
    {
        
        
        
        input_opencl_image    = (opencl_image_t *)malloc(sizeof(opencl_image_t));
        filtered_opencl_image = (opencl_image_t *)malloc(sizeof(opencl_image_t));
    
        /* Open input Image. */
        read_image = imageReadPPM(IMAGE_INPUT_FILENAME);
        
        /* Allocate pixels for RGBA image. */
        input_opencl_image->x     = read_image->x;
        input_opencl_image->y     = read_image->y;
        input_opencl_image->pixel = (opencl_pixel_t*)malloc(read_image->x * read_image->y * sizeof(opencl_pixel_t));
        
        filtered_opencl_image->x     = read_image->x;
        filtered_opencl_image->y     = read_image->y;
        filtered_opencl_image->pixel = (opencl_pixel_t*)malloc(read_image->x * read_image->y * sizeof(opencl_pixel_t));
        
        imageGetRGBAFromPPM(input_opencl_image, read_image);
    }
    
    /* Apply filter.
     */
    {
        cl_float threshold = 200.6f;
        
        /* Sobel filer. */
        cl_float filter [] =
        {
            -1, -2, -1,
             0, 0,   0,
             1, 2,   1
        };
        
        /* Apply filter on input image. */
        imageApplyFilter(filter, /* filter weights. */
                         threshold,
                         3,      /* filter nxn --> will be conculded inside this function. */
                         input_opencl_image, /* input image. */
                         filtered_opencl_image, /* output image. */
                         &err);
    }
    

    /* Save to PPM image.
     */
    {
        ppm_image_t *output_image;
        
        /* Allocate pixels for output image. */
        output_image = (ppm_image_t *)malloc(sizeof(ppm_image_t));
        
        output_image->x     = input_opencl_image->x;
        output_image->y     = input_opencl_image->y;
        output_image->pixel = (ppm_pixel_t *)malloc(input_opencl_image->x * input_opencl_image->y * sizeof(ppm_pixel_t));
        
        imageGetPPMFromRGBA(output_image, filtered_opencl_image);
        imageSavePPM(output_image, IMAGE_OUTPUT_FILENAME);
    }

    return 0;
}
