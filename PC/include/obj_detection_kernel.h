#ifndef _OBJ_DETECTION_KERNEL_H_
#define _OBJ_DETECTION_KERNEL_H_

#include "cuda_runtime.h"
#include "followPoint.h"




__global__ void mach_bmp_kernel(uint8_t *img_in,uint8_t *img_out, uint8_t *comp); 


#endif


