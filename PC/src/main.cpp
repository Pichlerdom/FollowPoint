
#include "followPoint.h"
#include "webcam_interface.h"
#include "follow_point_display.h"
#include "obj_detection_cuda.h"
#include <sys/time.h>

int main(int argc,const char* argv[]){
	
	Thread_Args *args = init_thread_args();
	
	pthread_t display_thread;
	pthread_t capture_thread;
	pthread_t detection_thread;
	
	pthread_create(	&display_thread, NULL, 
					main_display_loop, args);
	
	pthread_create( &capture_thread, NULL,
					main_capture_loop, args);

	pthread_create( &detection_thread, NULL,
					main_detection_loop, args);



	pthread_join(display_thread, NULL);
	pthread_join(capture_thread, NULL);
	pthread_join(detection_thread, NULL);
	return 0;
}


Thread_Args * init_thread_args(){
	Thread_Args *args = (Thread_Args *)calloc(1, sizeof(Thread_Args));
	
	pthread_mutex_init (&(args->image_buffer.read_mtx), NULL);
	pthread_mutex_init (&(args->image_buffer.proc_mtx), NULL);
	pthread_mutex_init (&(args->laser.mtx), NULL);
	pthread_mutex_init (&(args->control.mtx), NULL);


	args->image_buffer.read = (uint8_t *) calloc(WEBCAM_FRAME_WIDTH * WEBCAM_FRAME_HEIGHT * 3, sizeof(uint8_t)); 
	args->image_buffer.proc = (uint8_t *) calloc(WEBCAM_FRAME_WIDTH * WEBCAM_FRAME_HEIGHT * 3, sizeof(uint8_t)); 
	
	args->image_buffer.height = WEBCAM_FRAME_HEIGHT;
	args->image_buffer.width  = WEBCAM_FRAME_WIDTH;

	args->threshold = 0.001;

	args->comp = (uint8_t*) calloc(COMP_HEIGHT * COMP_WIDTH * 3, sizeof(uint8_t));
	
	args->control.capture	= true;
	args->control.display	= true;
	args->control.serial	= true;
	args->control.detection = true;
	
	return args;
}


void * main_capture_loop(void *arg){
	Thread_Args *args = (Thread_Args*) arg;
	uint8_t *frame_buffer;
	int frame_buffer_size = 0;

	int fd = open_camera(	"/dev/video0",
							&args->image_buffer.width, 
							&args->image_buffer.height, 
							&frame_buffer, &frame_buffer_size);

	struct timeval start, end;

	while(args->control.capture){
		gettimeofday(&start,NULL);

		fill_read_frame_rgb(fd, 
							frame_buffer, 
							args->image_buffer.read,	
							args->image_buffer.width,
							args->image_buffer.height);
		
		gettimeofday(&end, NULL);
		args->capturetime = ((end.tv_sec * 1000000 + end.tv_usec) - (start.tv_sec * 1000000 + start.tv_usec))/1000;
	}
	
	close_camera(fd);
	printf("Capture thread exited!\n");
	return NULL;
}
