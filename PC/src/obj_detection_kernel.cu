
#include<stdint.h>
#include<obj_detection_kernel.h>

__global__ void mach_bmp_kernel(uint8_t *img_in,uint8_t *img_out, uint8_t *comp){
	__shared__ float div[COMP_HEIGHT * COMP_WIDTH];

	unsigned int x = blockIdx.x + threadIdx.x;
	unsigned int y = blockIdx.y + threadIdx.y;
	
	unsigned int index =(gridDim.x * blockIdx.y + blockIdx.x);
	unsigned int pixIdx = (gridDim.x * y + x) * 3;
	unsigned int compIdx = blockDim.x * threadIdx.y + threadIdx.x;
	div[compIdx] = 200;
	if(	x < WEBCAM_FRAME_WIDTH &&
		y < WEBCAM_FRAME_HEIGHT){	
		float dr = fabsf((float)img_in[pixIdx + 0] - (float)comp[(compIdx * 3) + 0]);
		float dg = fabsf((float)img_in[pixIdx + 1] - (float)comp[(compIdx * 3) + 1]);
		float db = fabsf((float)img_in[pixIdx + 2] - (float)comp[(compIdx * 3) + 2]);
		div[compIdx] = (dr + db + dg)/3.0;

	}

	
	__syncthreads();
	for(unsigned int swap = (COMP_HEIGHT * COMP_WIDTH)/2; swap > 0;swap >>= 1){
		if(compIdx < swap){
			div[compIdx] += div[compIdx + swap];
		}
		__syncthreads();
	}

	__syncthreads();
	if(compIdx == 0){
		uint8_t gray = (uint8_t) (div[0] / (COMP_HEIGHT * COMP_WIDTH));

		img_out[index] = gray; 

	}
}
