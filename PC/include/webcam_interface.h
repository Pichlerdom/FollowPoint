#ifndef _WEBCAM_INTERFACE_H_
#define _WEBCAM_INTERFACE_H_
#include <stdint.h>

int open_camera(char *file_name, int *width, int *height,uint8_t **frame_buf, int *frame_buf_size);

void close_camera(int fd_cam);

void fill_frame_buf_yuyv(int fd_camera, uint8_t *frame_buf);

void fill_read_frame_rgb(int fd_camera, uint8_t *frame_buf, uint8_t *imgrgb, int width, int height);

void convert_YUYV_to_RGB(uint8_t *imgyuyv, uint8_t *imgrgb, int witdh, int height);

bool set_framerate(int fd,int framerate);

int xioctl(int fd, int request, void *arg);
#endif
