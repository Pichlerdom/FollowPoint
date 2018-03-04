


#include "obj_detection_cuda.h"

#include "obj_detection.h"

void cuda_test(ImgCont *imgCont,uint8_t *comp){
	uint8_t *d_img_in;
	uint8_t *d_img_out;
	uint8_t *d_comp;
	int imgsize = WEBCAM_FRAME_WIDTH * WEBCAM_FRAME_HEIGHT * 3;
	int compsize = COMP_WIDTH * COMP_HEIGHT * 3;

	uint8_t *img_out = (uint8_t*) malloc(imgsize/3);
	dim3 block(COMP_WIDTH,COMP_HEIGHT);
	dim3 grid(WEBCAM_FRAME_WIDTH/GRID_SIZE_X,WEBCAM_FRAME_HEIGHT/GRID_SIZE_Y);
	
	cudaMalloc(&d_img_in, imgsize);
	cudaMalloc(&d_img_out, imgsize/3);
	cudaMalloc(&d_comp, compsize);

	cudaMemcpy((void*)d_comp,
				   (void *)comp,
					compsize,
				   cudaMemcpyHostToDevice);

	cudaMemcpy((void*)d_img_in,
				   (void *)imgCont->img,
					imgsize,
				   cudaMemcpyHostToDevice);

	
	mach_bmp_kernel<<<grid,block>>>(d_img_in, d_img_out,d_comp);

	cudaMemcpy((void *)img_out,
			   (void *)d_img_out,
				imgsize/3,
			   cudaMemcpyDeviceToHost);

	find_delta(0, img_out, imgCont);
	/*f(	imgCont->laser.x != 0 &&
		imgCont->laser.y != 0){
		update_comp(imgCont);
	}*/
	free(img_out);
	cudaFree(d_img_out);
	cudaFree(d_comp);
	cudaFree(d_img_in);

}



void update_comp(ImgCont *imgCont){
	int foundx = imgCont->laser.x - (COMP_WIDTH/2);
	int foundy = imgCont->laser.y - (COMP_HEIGHT/2);
	for(int y = 0; y < COMP_HEIGHT; y++){
		for(int x = 0 ; x < COMP_WIDTH; x++){
			int compIdx = ((y * COMP_WIDTH) + x)*3;
			int imgIdx = (((y + foundy) * WEBCAM_FRAME_WIDTH) + foundx + x)*3;
			imgCont->comp[compIdx + 0] = imgCont->img[imgIdx + 0];
			imgCont->comp[compIdx + 1] = imgCont->img[imgIdx + 1];
			imgCont->comp[compIdx + 2] = imgCont->img[imgIdx + 2];
		}
	}
}

