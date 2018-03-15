#ifndef _OBJ_DETECTION_H_
#define _OBJ_DETECTION_H_

#include "followPoint.h"



int find_laser(float *img_div, uint8_t *img_disp, int *laserx, int *lasery, float threshold);

void blur(uint8_t *img);

#endif
