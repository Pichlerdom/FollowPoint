

#include <stdio.h>
#include "serial_interface.h"
#include "obj_detection.h"
#include <stdint.h>


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

			//	imgCont->img[(pixIdx * 3)  + 0] = 255;
			//	imgCont->img[(pixIdx * 3)  + 1] = 0;
			//	imgCont->img[(pixIdx * 3)  + 2] = 255;
			}
		}
	}
	if(count >0){
		int foundx =(int) sumx/count;
		int foundy =(int) sumy/count;
		int dy = -1*((foundx) - ((WEBCAM_FRAME_WIDTH/2)));
		int dx = (foundy) - ((WEBCAM_FRAME_HEIGHT/2));

		printf("laser at x: %d, y:%d \n",(sumx/count) ,(sumy/count));

		printf("delta of x: %d, y:%d \n",dx,dy);
		if(dx/5.0 > 7){
			imgCont->laser.dx = dx/5.0;
		}else{
			imgCont->laser.dx = 0;
		}
		imgCont->laser.dy = dy/10.0;
		imgCont->laser.x = (sumx/count) - COMP_WIDTH;
		imgCont->laser.y = (sumy/count) - COMP_HEIGHT;
	
	}else{
		imgCont->laser.dx = 0;
		imgCont->laser.dy = 0;
	}
}

