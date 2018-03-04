

#include <stdio.h>
#include "serial_interface.h"
#include "obj_detection.h"
#include <stdint.h>
#include <math.h>

#define THRESHOLD 15

void find_delta(int fd, uint8_t *img_out, ImgCont *imgCont){
	int count = 0;
	int sumx= 0;
	int sumy= 0;
	
	for(int y = 0; y < WEBCAM_FRAME_HEIGHT; y++){
		for(int x = 0; x < WEBCAM_FRAME_WIDTH; x++){
			int pixIdx = (y * WEBCAM_FRAME_WIDTH  + x);
			uint8_t gray = img_out[pixIdx];
			if(gray < THRESHOLD){
				count++;
				sumx += x;
				sumy += y;
			//printf("%d",gray);
				imgCont->img[(pixIdx * 3)  + 0] = 255;
				imgCont->img[(pixIdx * 3)  + 1] = 0;
				imgCont->img[(pixIdx * 3)  + 2] = 255;
			}
		}
	}
	if(count > 0){
		int foundx =(int) (sumx/count) + COMP_WIDTH/2;
		int foundy =(int) (sumy/count) + COMP_HEIGHT/2;
		int dx =  (WEBCAM_FRAME_WIDTH/2) - (foundx);
		int dy =  (WEBCAM_FRAME_HEIGHT/2) - (foundy);
		
		printf("laser at x: %d, y:%d \n",(sumx/count) ,(sumy/count));

		printf("delta of x: %d, y:%d \n",dx,dy);
		if(abs(dx) > 20){
			imgCont->laser.dx = dx;
		}
		if(abs(dy) > 10){
			imgCont->laser.dy = dy;
		}
		
		imgCont->laser.x = foundx;
		imgCont->laser.y = foundy;
		
	}else{
		imgCont->laser.dx = 0;
		imgCont->laser.dy = 0;
		
		imgCont->laser.x = 0;
		imgCont->laser.y = 0;
	}
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
