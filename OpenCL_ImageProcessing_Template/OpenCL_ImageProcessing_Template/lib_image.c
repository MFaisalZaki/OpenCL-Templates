
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>

#include "lib_opencl.h"
#include "lib_image.h"
//////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////


#define KERNEL_PRG_CNT 1
#define IMAGE_KERNEL_LIST_NAMES {"Filter"}
#define IMAGE_KERNEL_FILE_NAME "/Users/marwanfaisal/Desktop/OpenCL-Templates/OpenCL_ImageProcessing_Template/OpenCL_ImageProcessing_Template/kernel_filter.cl"

#define ERR_DEVICE_CONTEXT_CREATION_NOK 0
#define ERR_KERNEL_OBJS_CREATION_NOK    1
#define ERR_SIGNAL_OPERATION_NOK        2
#define ERR_BUFFER_CREATION_NOK         3
#define ERR_WRITE_BUFFER_NOK            4
#define ERR_SETTING_ARGUMENTS_NOK       5
#define ERR_READ_BUFFER_NOK             6

#define INFO_DEVICE_CONTEXT_CREATION_OK (ERR_DEVICE_CONTEXT_CREATION_NOK)
#define INFO_KERNEL_OBJS_CREATION_NOK   (ERR_KERNEL_OBJS_CREATION_NOK)

#define IMAGE_KERNEL_FILTER 0


static volatile cl_device_id * dev_list = NULL;
static cl_int     dev_cnt = 0;

static cl_context       image_context;
static cl_command_queue image_cmd_queue;
static cl_kernel        image_kernel_list[KERNEL_PRG_CNT];

static char * kernel_name_list[KERNEL_PRG_CNT] = IMAGE_KERNEL_LIST_NAMES;
//////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////

static void printImageInfoMsg(int msg_id);
static void printImageErrorMsg(int err_id);
//////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////

static void printImageInfoMsg(int msg_id)
{
    switch (msg_id)
    {
        case INFO_DEVICE_CONTEXT_CREATION_OK:
        {
            printf("Info Image processing component: Created device and context ... OK.\n");
            break;
        }
        case INFO_KERNEL_OBJS_CREATION_NOK:
        {
            printf("Info Image processing component: Create kernel objects ... OK\n");
            break;
        }
        default:
            break;
    }
}

static void printImageErrorMsg(int err_id)
{
    switch (err_id)
    {
        case ERR_DEVICE_CONTEXT_CREATION_NOK:
        {
            printf("Error Image processing component: Created device and context ... NOK.\n");
            break;
        }
        case ERR_KERNEL_OBJS_CREATION_NOK:
        {
            printf("Error Image processing component: Create kernel objects ... NOK.\n");
            break;
        }
        case ERR_BUFFER_CREATION_NOK:
        {
            printf("Error Image processing component: Create memory buffer for internal buffer ... NOK.\n");
            break;
        }
        case ERR_WRITE_BUFFER_NOK:
        {
            printf("Error Image processing component: Writing to buffer ... NOK.\n");
            break;
        }
        case ERR_SETTING_ARGUMENTS_NOK:
        {
            printf("Error Image processing component: Setting arguments for kernel ... NOK.\n");
            break;
        }
        case ERR_READ_BUFFER_NOK:
        {
            printf("Error Image processing component: Reading from buffer ... NOK.\n");
            break;
        }
        default:
            break;
    }
}
//////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////

void imageInit(const cl_device_id * const device_list,
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
                             &image_context,
                             &image_cmd_queue,
                             ret_err);
    
    if (*ret_err != CL_SUCCESS)
    {
        printImageInfoMsg(ERR_DEVICE_CONTEXT_CREATION_NOK);
        return;
    }
    else
    {
        printImageInfoMsg(INFO_DEVICE_CONTEXT_CREATION_OK);
    }
    
    /* Create program and kernel objects.
     */
    clCreateKernelObjsForContext(&image_context,
                                 (IMAGE_KERNEL_FILE_NAME),
                                 (const char **)kernel_name_list,
                                 (KERNEL_PRG_CNT),
                                 image_kernel_list,
                                 ret_err);
    if (*ret_err != CL_SUCCESS)
    {
        printImageErrorMsg(ERR_KERNEL_OBJS_CREATION_NOK);
    }
}

void imageApplyFilter(cl_float      filter[],
                      cl_int        size,
                      opencl_image_t * const input_image,
                      opencl_image_t * const ret_image,
                      cl_int         * const err)
{
    cl_mem input_image_buffer;
    cl_mem output_image_buffer;
    cl_mem filter_w_buffer;
    size_t global[3];
    
    /* Setup image description. */
    input_image_buffer = clCreateBuffer(image_context,
                                        (CL_MEM_READ_WRITE),
                                        (sizeof(opencl_pixel_t) * input_image->x * input_image->y),
                                        NULL,
                                        err);
    
    if (*err != CL_SUCCESS)
    {
        free(input_image_buffer);
        printImageErrorMsg(ERR_BUFFER_CREATION_NOK);
        return;
    }
    
    output_image_buffer = clCreateBuffer(image_context,
                                         (CL_MEM_READ_WRITE),
                                         (sizeof(opencl_pixel_t) * input_image->x * input_image->y),
                                         NULL,
                                         err);
    
    if (*err != CL_SUCCESS)
    {
        free(output_image_buffer);
        printImageErrorMsg(ERR_BUFFER_CREATION_NOK);
        return;
    }
    
    filter_w_buffer = clCreateBuffer(image_context,
                                     (CL_MEM_READ_ONLY),
                                      sizeof(cl_float) * (size*size),
                                     NULL,
                                     err);
    
    if (*err != CL_SUCCESS)
    {
        free(filter_w_buffer);
        printImageErrorMsg(ERR_BUFFER_CREATION_NOK);
        return;
    }
    /* Write input buffers to kernel. */
    *err = 0;

    /* Write image to kernel buffer. */
    *err = clEnqueueWriteBuffer(image_cmd_queue,
                                input_image_buffer,
                                CL_TRUE,
                                0,
                                (input_image->x * input_image->y * sizeof(opencl_pixel_t)),
                                (const void *)input_image->pixel,
                                0,
                                NULL,
                                NULL);

    /* Wait until copy is complete. */
    clFinish(image_cmd_queue);
    
    /* Write filter weights to kernel buffer. */
    *err |= clEnqueueWriteBuffer(image_cmd_queue,
                                 filter_w_buffer,
                                 CL_TRUE,
                                 0,
                                 (size * size * sizeof(cl_float)),
                                 (const void *)filter,
                                 0,
                                 NULL,
                                 NULL);
    
    /* Wait until copy is complete. */
    clFinish(image_cmd_queue);
    
    if (*err != CL_SUCCESS)
    {
        free(input_image_buffer);
        free(output_image_buffer);
        free(filter_w_buffer);
        
        printImageErrorMsg(ERR_WRITE_BUFFER_NOK);
        return;
    }
    
    *err = 0;
    
    /* Setup the kernel arguments. */
    *err |= clSetKernelArg(image_kernel_list[0], 0, sizeof (cl_mem), &input_image_buffer);
    *err |= clSetKernelArg(image_kernel_list[0], 1, sizeof (cl_mem), &output_image_buffer);
    *err |= clSetKernelArg(image_kernel_list[0], 2, sizeof (cl_mem), &filter_w_buffer);
    *err |= clSetKernelArg(image_kernel_list[0], 3, sizeof(cl_int), &size);
    
    if (*err != CL_SUCCESS)
    {
        free(input_image_buffer);
        free(output_image_buffer);
        free(filter_w_buffer);
        
        printImageErrorMsg(ERR_SETTING_ARGUMENTS_NOK);
        return;
    }
    
    /* Set global parameters. */
    global[0] = input_image->x;
    global[1] = input_image->y;
    global[2] = 1;
    
    /* Execute command queue. */
    *err = clEnqueueNDRangeKernel(image_cmd_queue,
                                  image_kernel_list[IMAGE_KERNEL_FILTER],
                                  2, /* 2-Dim. */
                                  NULL,
                                  global,
                                  NULL,
                                  0,
                                  NULL,
                                  NULL);
    
    /* Wait for command queue to finish. */
    clFinish(image_cmd_queue);
    
    /* Read output buffer. */
    *err |= clEnqueueReadBuffer(image_cmd_queue,
                               output_image_buffer,
                               CL_TRUE,
                               0,
                               (input_image->x * input_image->y * sizeof(opencl_pixel_t)),
                               (void *)ret_image->pixel,
                               0,
                               NULL,
                               NULL);
    
    /* Wait for read to be complete. */
    clFinish(image_cmd_queue);

    if (*err != CL_SUCCESS)
    {
        printImageErrorMsg(ERR_READ_BUFFER_NOK);
        return;
    }
    
    /* Clean buffers. */
    clReleaseMemObject(input_image_buffer);
    clReleaseMemObject(filter_w_buffer);
    clReleaseMemObject(output_image_buffer);
    
    *err = CL_SUCCESS;
}

void imageGetRGBAFromPPM(opencl_image_t * const ret_image,
                         ppm_image_t    * const ppm_image)
{
    
    for(cl_int i = 0; i < (ppm_image->x*ppm_image->y); i += 1)
    {
        ret_image->pixel[i].blue  = (cl_float)ppm_image->pixel[i].blue;
        ret_image->pixel[i].green = (cl_float)ppm_image->pixel[i].green;
        ret_image->pixel[i].red   = (cl_float)ppm_image->pixel[i].red;
        ret_image->pixel[i].alpha = (cl_float)0;
    }
}

void imageGetPPMFromRGBA(ppm_image_t * const ret_image,
                         opencl_image_t * const rgba_image)
{
    
    for(cl_int i = 0; i < (rgba_image->x*rgba_image->y); i += 1)
    {
        ret_image->pixel[i].blue  = (unsigned char)rgba_image->pixel[i].blue;
        ret_image->pixel[i].green = (unsigned char)rgba_image->pixel[i].green;
        ret_image->pixel[i].red   = (unsigned char)rgba_image->pixel[i].red;
    }
}

void imageSavePPM(ppm_image_t * const input_image,
                  const char * const output_image_filename)
{
    FILE *image_file_ptr;
    
    image_file_ptr = fopen(output_image_filename, "wb");
    
    /* Open image output file. */
    if (!image_file_ptr)
    {
        fprintf(stderr, "Unable to open file '%s'\n", output_image_filename);
        exit(1);
    }
    
    /* Write the header file for output image. */
    fprintf(image_file_ptr, "P6\n");
    
    /* Write comments. */
    fprintf(image_file_ptr, "# Output image creation.\n");
    
    /* Write image size. */
    fprintf(image_file_ptr, "%d %d\n", input_image->x, input_image->y);
    
    /* Write RGB component depth. */
    fprintf(image_file_ptr, "%d\n", RGB_COMPONENT_COLOR);
    
    /* Write pixels. */
    fwrite(input_image->pixel, (3*input_image->x), input_image->y, image_file_ptr);
    
    /* Close file. */
    fclose(image_file_ptr);
}

ppm_image_t * imageReadPPM(const char * const image_filename)
{
    char buffer[16];
    FILE *image_file_ptr;
    ppm_image_t *ret_image;
    int c;
    int rgb_cmp_color;
    
    /* Open image. */
    image_file_ptr = fopen(image_filename, "rb");
    
    /* Allocate image. */
    ret_image = (ppm_image_t *)malloc(sizeof(ppm_image_t));
    
    /* Try to open image. */
    if (!image_file_ptr)
    {
        fprintf(stderr, "Unable to open file %s", image_filename);
        exit(1);
    }
    /* Check if buffer out of size. */
    else if(!fgets(buffer, sizeof(buffer), image_file_ptr))
    {
        perror(image_filename);
        exit(1);
    }
    /* Check if image is allocated. */
    else if (!ret_image)
    {
        fprintf(stderr, "Unable to allocate memory\n");
        exit(1);
    }
    /* Check image format. */
    else if((buffer[0] != 'P') || (buffer[1] != '6'))
    {
        fprintf(stderr, "Invalid image format (must be 'P6')\n");
        exit(1);
    }
    else
    {
        /* Do nothing. */
    }
    
    
    /* Check for comments. */
    c = getc(image_file_ptr);
    while (c == '#')
    {
        while (getc(image_file_ptr) != '\n');
        c = getc(image_file_ptr);
    }
    ungetc(c, image_file_ptr);
    
    /* Check on image size information. */
    if (fscanf(image_file_ptr, "%d %d", &ret_image->x, &ret_image->y) != 2) {
        fprintf(stderr, "Invalid image size (error loading '%s')\n", image_filename);
        exit(1);
    }
    /* Read RGB component. */
    else if (fscanf(image_file_ptr, "%d", &rgb_cmp_color) != 1) {
        fprintf(stderr, "Invalid rgb component (error loading '%s')\n", image_filename);
        exit(1);
    }
    else
    {
        /* Do nothing. */
    }
    
    /* Check RBG component depth */
    if (rgb_cmp_color!= RGB_COMPONENT_COLOR) {
        fprintf(stderr, "'%s' does not have 8-bits components\n", image_filename);
        exit(1);
    }
    
    while (fgetc(image_file_ptr) != '\n') ;
    
    /* memory allocation for pixel data. */
    ret_image->pixel = (ppm_pixel_t*)malloc(ret_image->x * ret_image->y * sizeof(ppm_pixel_t));
    
    if (!ret_image) {
        fprintf(stderr, "Unable to allocate memory\n");
        exit(1);
    }
    
    /* read pixel data from file. */
    if (fread(ret_image->pixel, (3 * ret_image->x), ret_image->y, image_file_ptr) != ret_image->y) {
        fprintf(stderr, "Error loading image '%s'\n", image_filename);
        exit(1);
    }
    
    fclose(image_file_ptr);
    return (ret_image);
}
