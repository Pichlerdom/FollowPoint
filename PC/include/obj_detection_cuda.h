#ifndef _OBJ_DETECTION_CUDA_H_
#define _OBJ_DETECTION_CUDA_H_

#include "obj_detection_kernel.h"
#include "obj_detection.h"
#include "cuda_runtime.h"
#include <stdint.h>
#include "followPoint.h"

void * main_detection_loop(void * arg);

void update_comp(uint8_t *comp, uint8_t *img,  int laserx , int lasery);

#endif
