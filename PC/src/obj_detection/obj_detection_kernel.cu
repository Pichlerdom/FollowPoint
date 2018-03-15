
#include<stdint.h>
#include<obj_detection_kernel.h>

__global__ void mach_bmp_kernel(uint8_t *img_in, float *img_out, uint8_t *comp){
	__shared__ float div[COMP_HEIGHT * COMP_WIDTH];

	unsigned int x = blockIdx.x + threadIdx.x;
	unsigned int y = blockIdx.y + threadIdx.y;
	
	unsigned int index =(gridDim.x * blockIdx.y + blockIdx.x);
	unsigned int pixIdx = (gridDim.x * y + x) * 3;
	unsigned int compIdx = blockDim.x * threadIdx.y + threadIdx.x;
	div[compIdx] = 200;
	if(	x < WEBCAM_FRAME_WIDTH &&
		y < WEBCAM_FRAME_HEIGHT){	
		float r = (float)img_in[pixIdx + 0];
		float g = (float)img_in[pixIdx + 1];
		float b = (float)img_in[pixIdx + 2];
		float gray = (r+g+b)/3.0;
		float dr = fabsf(r - (float)comp[(compIdx * 3) + 0]);
		float dg = fabsf(g - (float)comp[(compIdx * 3) + 1]);
		float db = fabsf(b - (float)comp[(compIdx * 3) + 2]);
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
		float gray = (div[0] / (COMP_HEIGHT * COMP_WIDTH));

		img_out[index] = (float) gray * 0.9; 

	}
}

__global__ void mach_bmp_kernel_gray(uint8_t *img_in,float *img_out, uint8_t *comp){
	__shared__ float div[COMP_HEIGHT * COMP_WIDTH];

	unsigned int x = blockIdx.x + threadIdx.x;
	unsigned int y = blockIdx.y + threadIdx.y;
	
	unsigned int index =(gridDim.x * blockIdx.y + blockIdx.x);
	unsigned int pixIdx = (gridDim.x * y + x);
	unsigned int compIdx = blockDim.x * threadIdx.y + threadIdx.x;

	float img_pix = img_in[pixIdx];
	float comp_pix = comp[compIdx];

	div[compIdx] = fabsf(img_pix - comp_pix);

	__syncthreads();
	for(unsigned int swap = (COMP_HEIGHT * COMP_WIDTH)/2; swap > 0;swap >>= 1){
		if(compIdx < swap){
			div[compIdx] += div[compIdx + swap];
		}
		__syncthreads();
	}

	if(compIdx == 0){
		float gray = (div[0] / (float)(COMP_HEIGHT * COMP_WIDTH));

		img_out[index] +=(float) gray * 1.1; 

	}
}

__global__ void laplace_edge_detection(uint8_t *img_in, uint8_t *img_out, int8_t *matrix){
	__shared__ float result[9];
	
	unsigned int x = blockIdx.x + threadIdx.x;
	unsigned int y = blockIdx.y + threadIdx.y;
	unsigned int img_in_index = (gridDim.x * y + x) * 3; 
	unsigned int matrix_index = threadIdx.y * blockDim.x + threadIdx.x;

	float r = (float) img_in[img_in_index + 0];
	float g = (float) img_in[img_in_index + 1];
	float b = (float) img_in[img_in_index + 2]; 
	float m = (float) matrix[matrix_index];

	float gray = (r + g + b)/3.0;
	result[matrix_index] = ((float)m * gray); 
	
	__syncthreads();
	if(matrix_index == 0){
		
		float sum = 0;
		for(int i= 0; i < 9;i++){
			sum += result[i];
		}
		img_out[blockIdx.y * gridDim.x + blockIdx.x] = (uint8_t)sum;
	}
}

__global__ void threshold_filter_gray(uint8_t *img_in, uint8_t *img_out, float threshold){
	unsigned int index = gridDim.x * blockIdx.x + threadIdx.x;

	if(img_in[index] > threshold){
		img_out[index] = 255;
	}else{
		img_out[index] = 0;
	}
	__syncthreads();
}

__global__ void threshold_filter_rgb(uint8_t *img_in, uint8_t *img_out, float threshold){
	unsigned int index = (gridDim.x * blockIdx.x + threadIdx.x) * 3;
	float r = img_in[index];
	float g = img_in[index + 1];
	float b = img_in[index + 2];
	float gray = (r+b+g)/3.0;

	if(gray > threshold){
		img_out[index] = 255;
	}else{
		img_out[index] = 0;
	}
	__syncthreads();
}

__global__ void blur_rgb(uint8_t *img_in, uint8_t *img_out){
	__shared__ uint8_t result[9 * 3];
	
	unsigned int x = blockIdx.x + threadIdx.x;
	unsigned int y = blockIdx.y + threadIdx.y;
	unsigned int img_in_index = (gridDim.x * y + x) * 3; 
	unsigned int matrix_index = (threadIdx.y * blockDim.x + threadIdx.x) * 3;

	result[matrix_index + 0] = (float) img_in[img_in_index + 0];
	result[matrix_index + 1] = (float) img_in[img_in_index + 1];
	result[matrix_index + 2] = (float) img_in[img_in_index + 2]; 
	
	__syncthreads();
	if(matrix_index == 0){
		
		float sumr = 0;
		float sumg = 0;
		float sumb = 0;
		for(int i= 0; i < 9 * 3; i+=3){
			sumr += (float)result[i + 0];
			sumg += (float)result[i + 1];
			sumb += (float)result[i + 2];

		}
		img_out[img_in_index] = (uint8_t)(sumr/9.0);
		img_out[img_in_index + 1] = (uint8_t)(sumg/9.0);
		img_out[img_in_index + 2] = (uint8_t)(sumb/9.0);
	
	}

}
