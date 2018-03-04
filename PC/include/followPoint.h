#ifndef _FOLLOW_POINT_H_
#define _FOLLOW_POINT_H_


#include <SDL2/SDL.h>
#include <SDL2/SDL_image.h>

#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <stdint.h>
#include <unistd.h>
#include <fcntl.h>
#include <string.h>

#define WINDOW_WIDTH WEBCAM_FRAME_WIDTH + COMP_WIDTH 
#define WINDOW_HEIGHT WEBCAM_FRAME_HEIGHT
#define COMP_HEIGHT 16
#define COMP_WIDTH 16

#define GRID_SIZE_X 1
#define GRID_SIZE_Y 1

#define WEBCAM_FRAME_WIDTH 640
#define WEBCAM_FRAME_HEIGHT 480

typedef struct{
	uint8_t *img;
	uint8_t *comp;
	int width;
	int height;
	pthread_mutex_t imgMtx;
	struct{
		int dx;
		int dy;
		int x;
		int y;
	}laser;
	struct {
		bool capture;
		bool display;
		bool detection;
	}control;
}ImgCont;

#endif
