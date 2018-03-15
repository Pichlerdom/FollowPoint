#ifndef _FOLLOW_POINT_H_
#define _FOLLOW_POINT_H_

#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <stdint.h>
#include <unistd.h>
#include <fcntl.h>
#include <string.h>

#define WINDOW_WIDTH WEBCAM_FRAME_WIDTH + WEBCAM_FRAME_WIDTH
#define WINDOW_HEIGHT WEBCAM_FRAME_HEIGHT  + COMP_WIDTH 
#define COMP_HEIGHT 16
#define COMP_WIDTH 16

#define GRID_SIZE_X 1
#define GRID_SIZE_Y 1

#define WEBCAM_FRAME_WIDTH  640
#define WEBCAM_FRAME_HEIGHT 480


typedef struct{
	struct{		
		pthread_mutex_t read_mtx;
		uint8_t *read;
		pthread_mutex_t proc_mtx;
		uint8_t *proc;

		int width;
		int height;
	}image_buffer;
	float threshold;
	uint8_t *comp;
	pthread_mutex_t comp_mtx;
	struct{
		pthread_mutex_t mtx;
		int dx;
		int dy;
		int x;
		int y;
		unsigned int pixcount;
	}laser;
	uint32_t calctime;
	uint32_t capturetime;
	struct {
		pthread_mutex_t mtx;
		bool capture;
		bool display;
		bool serial;
		bool detection;
	}control;
}Thread_Args;



void * main_capture_loop(void *arg);

Thread_Args * init_thread_args();

#endif
