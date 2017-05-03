#ifndef _LIB_IMAGE_H_
#define _LIB_IMAGE_H_

#include <OpenCL/OpenCL.h>

#define RGB_COMPONENT_COLOR 255

typedef struct {
    unsigned char red;
    unsigned char green;
    unsigned char blue;
}ppm_pixel_t;

typedef struct {
    int x;
    int y;
    ppm_pixel_t *pixel;
}ppm_image_t;

typedef struct {
    cl_float red;
    cl_float green;
    cl_float blue;
    cl_float alpha;
}opencl_pixel_t;

typedef struct {
    int x;
    int y;
    opencl_pixel_t *pixel;
}opencl_image_t;

extern void imageInit(const cl_device_id * const device_list,
                            cl_int               num_dev,
                            cl_int       * const ret_err);


extern void imageApplyFilter(cl_float      filter[],
                             cl_int        size,
                             opencl_image_t * const input_image,
                             opencl_image_t * const ret_image,
                             cl_int         * const err);

extern void imageGetRGBAFromPPM(opencl_image_t * const ret_image,
                                ppm_image_t    * const ppm_image);

extern void imageGetPPMFromRGBA(ppm_image_t * const ret_image,
                                opencl_image_t * const rgba_image);

extern void imageSavePPM(ppm_image_t * const input_image,
                         const char * const output_image_filename);

extern ppm_image_t * imageReadPPM(const char * const image_filename);

#endif /* _LIB_IMAGE_H_ */
