#ifndef _OBJ_DETECTION_CUDA_H_
#define _OBJ_DETECTION_CUDA_H_

#include "obj_detection_kernel.h"
#include "obj_detection.h"
#include "cuda_runtime.h"
#include <stdint.h>
#include "followPoint.h"

void cuda_test(ImgCont *imgCont,uint8_t *comp);

void update_comp(ImgCont *imgCont);
#endif
