
#include "followPoint.h"
#include "webcam_interface.h"
#include "follow_point_display.h"

int main(int argc,const char* argv[]){

	ImgCont *imgCont = (ImgCont *) calloc(1, sizeof(ImgCont));

	imgCont->width  = WEBCAM_FRAME_WIDTH;
	imgCont->height = WEBCAM_FRAME_HEIGHT;

	imgCont->control.capture	= true;
	imgCont->control.display	= true;
	imgCont->control.detection	= false;
	imgCont->comp = NULL;

	imgCont->img = (uint8_t *) calloc(WEBCAM_FRAME_WIDTH * WEBCAM_FRAME_HEIGHT * 3, sizeof(uint8_t));
	
	pthread_t display_t, capture_t;
	
	//start the display thread
	pthread_create(&display_t, NULL, &main_display_loop, imgCont);
	pthread_create(&capture_t, NULL, &main_capture_loop, imgCont);

	pthread_join(display_t, NULL);
	printf("Display thread joined! \n");
	
	imgCont->control.capture = false;
	pthread_join(capture_t, NULL);	
	printf("Capture thread joined! \n");

	free(imgCont);
	return 0;





}
