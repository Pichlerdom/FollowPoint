#ifndef _OBJ_DETECTION_KERNEL_H_
#define _OBJ_DETECTION_KERNEL_H_

#include "cuda_runtime.h"
#include "followPoint.h"




__global__ void mach_bmp_kernel(uint8_t *img_in,float *img_out, uint8_t *comp); 
__global__ void mach_bmp_kernel_gray(uint8_t *img_in,float *img_out, uint8_t *comp);


__global__ void threshold_filter_gray(uint8_t *img_in, uint8_t *img_out, float threshold);
__global__ void threshold_filter_rgb(uint8_t *img_in, uint8_t *img_out, float threshold);
__global__ void laplace_edge_detection(uint8_t *img_in, uint8_t *img_out, int8_t *matrix);

__global__ void blur_rgb(uint8_t *img_in, uint8_t *img_out);
#endif


