


#include "obj_detection_cuda.h"
#include "obj_detection.h"
#include "serial_interface.h"
#include <sys/time.h>

#define OPT_COUNT 3

void * main_detection_loop(void * arg){
	Thread_Args *args = (Thread_Args*)arg;
	
	int fd = open_com_port("/dev/ttyUSB0");

	int imgsize = WEBCAM_FRAME_WIDTH * WEBCAM_FRAME_HEIGHT * 3;
	int compsize = COMP_WIDTH * COMP_HEIGHT * 3;


	uint8_t *d_img_in;
	float *d_img_div;
	uint8_t *d_img_edge;
	uint8_t *d_comp;
	uint8_t *d_edge;
	int8_t matrix[9] = {0,1,0,
						1,-4,1,
						0,1,0};
	int8_t *d_matrix;

	float *img_div = (float*) malloc(imgsize/3 * sizeof(float));



	dim3 block(COMP_WIDTH,COMP_HEIGHT);
	dim3 grid(WEBCAM_FRAME_WIDTH/GRID_SIZE_X,WEBCAM_FRAME_HEIGHT/GRID_SIZE_Y);
	dim3 laplaceblock(3,3);
	dim3 compgrid(COMP_WIDTH, COMP_HEIGHT);

	cudaMalloc(&d_img_in, imgsize);
	cudaMalloc(&d_img_div, imgsize/3 * sizeof(float));
	cudaMalloc(&d_comp, compsize);
	cudaMalloc(&d_edge, compsize/3);
	cudaMalloc(&d_img_edge, imgsize/3);
	
	int matrix_size = 9;
	cudaMalloc(&d_matrix, matrix_size);

	struct timeval start, end;
	long int frametime = 0;
	cudaMemcpy((void *)d_matrix, (void *)matrix,
				9,
				cudaMemcpyHostToDevice);

	cudaMemcpy((void*)d_comp,
				   (void *)args->comp,
					compsize,
				   cudaMemcpyHostToDevice);


	int8_t serial_buffer[2];

	int framecount = 0;
	int pixcount = 0;
	while(args->control.detection){
		gettimeofday(&start,NULL);
		
		cudaMemcpy((void*)d_img_in,
					   (void *)args->image_buffer.read,
						imgsize,
					   cudaMemcpyHostToDevice);

					

			cudaMemcpy((void*)d_comp,
					   (void *)args->comp,
						compsize,
					   cudaMemcpyHostToDevice);

		//blur_rgb<<<grid,laplaceblock>>>(d_img_in, d_img_in);

		//blur_rgb<<<compgrid,laplaceblock>>>(d_comp, d_comp);

		laplace_edge_detection<<<compgrid,laplaceblock>>>(d_comp, d_edge, d_matrix);
	
		laplace_edge_detection<<<grid,laplaceblock>>>(d_img_in, d_img_edge, d_matrix);

		mach_bmp_kernel<<<grid,block>>>(d_img_in, d_img_div,d_comp );
		mach_bmp_kernel_gray<<<grid,block>>>(d_img_edge, d_img_div, d_edge);

		cudaMemcpy((void *)img_div,
				   (void *)d_img_div,
					imgsize/3 * sizeof(float),
				   cudaMemcpyDeviceToHost);

		
		pixcount = find_laser(	img_div,
								args->image_buffer.proc,
								&(args->laser.x),
								&(args->laser.y),
								args->threshold);

		args->laser.pixcount = pixcount;

		if(pixcount < OPT_COUNT){
			if(pixcount <= 1){
				args->threshold += 0.2;
			}else{
				args->threshold += 0.01;
			}
			if(args->threshold > 100.0){
				args->threshold = 100.0;
			}
		}else if(pixcount > OPT_COUNT){
			if(pixcount > 30){
				args->threshold -= 0.2;
			}else{
				args->threshold -= 0.01;
			}
		}
		if(pixcount > 0 && pixcount < 50){
			if(framecount % 3 == 0){
				if(args->laser.x != 0){
					args->laser.dx = args->laser.x - (WEBCAM_FRAME_WIDTH / 2);
				}else{
					args->laser.dx = 0;
				} 
				if(args->laser.y != 0){
					args->laser.dy = args->laser.y - (WEBCAM_FRAME_HEIGHT / 2);
				}else{
					args->laser.dy = 0;
				}
				int motorspeed = 40;
				if(abs(args->laser.dx) > 200){
					motorspeed = 100;
				}
				if(args->laser.dx > 20){
						serial_buffer[0] = (uint8_t)motorspeed;
				}else if(args->laser.dx < -20){
					serial_buffer[0] = (uint8_t)-motorspeed;
				}else{
					serial_buffer[0] = 0;
				}
				int8_t servodelta = 1;
				if(abs(args->laser.dy) > 100){
					servodelta = 2; 
				}
				

				if(args->laser.dy < -20){
					serial_buffer[1] = -servodelta;
				}else if(args->laser.dy > 20){
					serial_buffer[1] = servodelta;
				}else{
					serial_buffer[1] = 0;
				}
			}else{
				serial_buffer[1] =0;
			}

			if(	(abs(serial_buffer[0]) > 0 ||
				abs(serial_buffer[1]) > 0) && fd >= 0){
				
				printf("sending x:y -> %d, %d\n",serial_buffer[0], serial_buffer[1]);
				write_bytes(fd, (char*)serial_buffer, 2);
			}
		}
	

		gettimeofday(&end, NULL);
		frametime = ((end.tv_sec * 1000000 + end.tv_usec) - (start.tv_sec * 1000000 + start.tv_usec));
		args->calctime = frametime/1000;

		framecount++;
	}

	free(img_div);
	cudaFree(d_img_div);
	cudaFree(d_comp);
	cudaFree(d_matrix);
	cudaFree(d_img_in);
	close_com_port(fd);
	printf("Detection thread exited!\n");
	return NULL;
}



void update_comp(uint8_t *comp, uint8_t *img,  int laserx , int lasery){
	int foundx = laserx - (COMP_WIDTH/2);
	int foundy = lasery - (COMP_HEIGHT/2);
	for(int y = 0; y < COMP_HEIGHT; y++){
		for(int x = 0 ; x < COMP_WIDTH; x++){
			int compIdx = ((y * COMP_WIDTH) + x)*3;
			int imgIdx = (((y + foundy) * WEBCAM_FRAME_WIDTH) + foundx + x)*3;
			comp[compIdx + 0] = img[imgIdx + 0];
			comp[compIdx + 1] = img[imgIdx + 1];
			comp[compIdx + 2] = img[imgIdx + 2];
		}
	}
}

