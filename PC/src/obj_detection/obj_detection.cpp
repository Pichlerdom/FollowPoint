

#include <stdio.h>
#include "serial_interface.h"
#include "obj_detection.h"
#include <stdint.h>
#include <math.h>

#define THRESHOLD 0

int find_laser(float *img_div, uint8_t *img_disp, int *laserx, int *lasery, float threshold){
	int count = 0;
	int sumx= 0;
	int sumy= 0;
	
	for(int y = 0; y < WEBCAM_FRAME_HEIGHT - COMP_HEIGHT; y++){
		for(int x = 0; x < WEBCAM_FRAME_WIDTH - COMP_WIDTH; x++){
			int pixIdx = (y * WEBCAM_FRAME_WIDTH  + x);
			float gray = img_div[pixIdx];
			if(gray < threshold){
				count++;
				sumx += x;
				sumy += y;
				img_disp[pixIdx] = 0;
			}else{
			
				img_disp[pixIdx] = (uint8_t)gray;
			}
		}
	}
	if(count > 0){
		*laserx =(int) (sumx/count) + COMP_WIDTH/2;
		*lasery =(int) (sumy/count) + COMP_HEIGHT/2;
		
	}else{
		*laserx = 0;
		*lasery = 0;
	}
	return count;
}


void blur(uint8_t *img){
	for(int y = 1; y < WEBCAM_FRAME_HEIGHT- 1; y++){
		for(int x = 1; x < WEBCAM_FRAME_WIDTH - 1; x++) {
			int pixIdx = (y * WEBCAM_FRAME_WIDTH + x) * 3;
			int sumrgb[] = {0,0,0};
			for(int dx = -1; dx < 2; dx++){
				for(int dy = -1; dy < 2; dy++){
					int pixIdx2 = ((y + dy) * WEBCAM_FRAME_WIDTH + x + dx) * 3;
			
					sumrgb[0] += img[pixIdx2];
					sumrgb[1] += img[pixIdx2 + 1];
					sumrgb[2] += img[pixIdx2 + 2];

				}
			}
			img[pixIdx] = sumrgb[0] / 9.0;
			img[pixIdx+1] = sumrgb[1] / 9.0;
			img[pixIdx+2] = sumrgb[2] / 9.0;
			sumrgb[0] = 0;
			sumrgb[1] = 0;
			sumrgb[2] = 0;

		}
	}
}
