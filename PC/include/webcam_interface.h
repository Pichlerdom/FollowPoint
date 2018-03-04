
#ifndef _WEBCAM_INTERFACE_H_
#define _WEBCAM_INTERFACE_H_

#include "followPoint.h"

#define DEFAULT_VIDEO_DEVICE "/dev/video1"


static int xioctl(int fd, int request, void *arg);

int allocCamera(ImgCont *imgCont, uint8_t ** framebuf);

void closeCamera(int fd_cam);

void *main_capture_loop(void* args);

void readframe(ImgCont *imgCont, int fd_camera, uint8_t *buf);

void convert_YUYV_to_RGB(uint8_t *imgyuyv, uint8_t *imgrgb, int witdh, int height);

void writeframebuf(uint8_t *framebuf, ImgCont *imgCont);

#endif

