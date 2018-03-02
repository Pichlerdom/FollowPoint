

#include "followPoint.h"
#include "follow_point_display.h"
#include "webcam_interface.h"

void* main_display_loop(void *arg){
	
	ImgCont *imgCont = (ImgCont *) arg;	

	Display *display = init_display();
	SDL_Event e;
	bool mousekeydown = false;
	uint32_t currTime = SDL_GetTicks();
	uint32_t frameTime = 0u;
	
	int mousex = 0;
	int mousey = 0;

	while(imgCont->control.display){
		currTime = SDL_GetTicks();		
		
		SDL_GetMouseState(&mousex, &mousey);
		//event handling loop
		while(SDL_PollEvent(&e) != 0){

			switch(e.type){
			
				//close application
				case SDL_QUIT:
					imgCont->control.display = false;
				break;
				case SDL_MOUSEBUTTONUP:
					imgCont->comp = (uint8_t*) calloc(COMP_HEIGHT * COMP_WIDTH * 3,sizeof(uint8_t));
					for(int dy = 0; dy < COMP_HEIGHT; dy++){
						for(int dx = 0; dx < COMP_WIDTH; dx++){
							if(mousex + dx < WINDOW_WIDTH && mousey + dy < WINDOW_HEIGHT) {
								int pixIdx = (WEBCAM_FRAME_WIDTH * (mousey + dy) + (mousex + dx)) * 3 ;
								int compIdx = (COMP_WIDTH * dy + dx) * 3;
								imgCont->comp[compIdx] = imgCont->img[pixIdx];
								imgCont->comp[compIdx + 1] = imgCont->img[pixIdx + 1];
								imgCont->comp[compIdx + 2] = imgCont->img[pixIdx + 2];
							}else{
							
							}
						}
					}
					imgCont->control.detection = true;
					mousekeydown = false;
				break;
				case SDL_MOUSEBUTTONDOWN:
				
					imgCont->control.detection = false;
					free(imgCont->comp);
					imgCont->comp = NULL;
					mousekeydown = true;
				break;
			}
		}

		

		//Clear screen
		SDL_SetRenderDrawColor( display->renderer, 0x00, 0x00, 0x00, 0xFF );
		SDL_RenderClear( display->renderer );
		pthread_mutex_lock(&(imgCont->imgMtx));
		if(imgCont->img != NULL){
			draw_img(display, imgCont->img,
					 WEBCAM_FRAME_WIDTH, WEBCAM_FRAME_HEIGHT,
					 0, 0);
		}
		if(imgCont->comp != NULL){
			draw_img(display, imgCont->comp,
					 COMP_HEIGHT, COMP_WIDTH,
					 WEBCAM_FRAME_WIDTH,0);
		}

		if(mousekeydown){

			SDL_Rect rect;
			rect.x = mousex;
			rect.y = mousey;
			rect.h = COMP_HEIGHT;
			rect.w = COMP_WIDTH;

			SDL_SetRenderDrawColor( display->renderer, 0xFF, 0x00, 0xFF, 0xFF);
			SDL_RenderDrawRect(display->renderer, &rect);
		
		}
		pthread_mutex_unlock(&(imgCont->imgMtx));
		//Update screen
		SDL_RenderPresent( display->renderer );

		//FPS stuff
		frameTime = SDL_GetTicks() - currTime;
		
		//printf("disp:%d\n", frameTime);
		if(frameTime > MS_PER_FRAME){
			frameTime = MS_PER_FRAME;
		}
		SDL_Delay(MS_PER_FRAME-frameTime);

	}
	
	close(display);
	return NULL;

}
void draw_circle_at(Display* display, int radius, int xpos, int ypos){
	for (int w = 0; w < radius * 2; w++)
	{
		for (int h = 0; h < radius * 2; h++)
		{
			int dx = radius - w; // horizontal offset
			int dy = radius - h; // vertical offset
			if ((dx*dx + dy*dy) <= (radius * radius))
			{
				SDL_RenderDrawPoint(display->renderer, xpos + dx, ypos + dy);
			}
		}
	}
}

void draw_img(Display* display,uint8_t *img, int width, int height, int xpos, int ypos){
	for(int y = 0;  y < height; y++){
		for(int x = 0; x < width; x++){
			int pixIdx = (width * y + x) * 3;
			uint8_t r = img[pixIdx];
			uint8_t g = img[pixIdx + 1];
			uint8_t b = img[pixIdx + 2];
			SDL_SetRenderDrawColor(display->renderer, r, g, b, 0xff);
			SDL_RenderDrawPoint(display->renderer, x + xpos, y + ypos);
		}
	}
	

	
}
void draw_img_grey(Display* display,uint8_t *img, int width, int height, int xpos, int ypos){
	for(int y = 0;  y < height; y++){
		for(int x = 0; x < width; x++){

			int pixIdx = (width * y + x);
			uint8_t g = img[pixIdx];
			
			SDL_SetRenderDrawColor(display->renderer, g, g, g, 0xff);
			SDL_RenderDrawPoint(display->renderer, x + xpos, y + ypos);
		}
	}
	

	
}

Display* init_display(){ 
	Display *display = (Display *) calloc(1, sizeof(Display));
	if(display == NULL){
		printf("Display could not be created!");
		return NULL;
	}
	//Initialize SDL
	if( SDL_Init( SDL_INIT_VIDEO ) < 0 )
	{
		printf( "SDL could not initialize! SDL_Error: %s\n", SDL_GetError() );
	    
        return NULL;
	}
	
	IMG_Init(IMG_INIT_JPG);

    display->window = SDL_CreateWindow("followPoint",
                                       SDL_WINDOWPOS_UNDEFINED,
                                       SDL_WINDOWPOS_UNDEFINED,
                                       WINDOW_WIDTH,
                                       WINDOW_HEIGHT,
                                       SDL_WINDOW_SHOWN);
    if(display->window == NULL) {
        printf("Unable to create Window! SDL_Error: %s\n", SDL_GetError());
        return NULL;
    }
	display->renderer = SDL_CreateRenderer( display->window, -1, SDL_RENDERER_ACCELERATED );
	if( display->renderer == NULL )
	{
		printf( "Renderer could not be created! SDL Error: %s\n", SDL_GetError() );
		return NULL;
	}

	//Initialize renderer color
	SDL_SetRenderDrawColor(display->renderer, 0xFF, 0xFF, 0xFF, 0xFF ); 

    return display;
 
}

void close(Display* display){
    
	SDL_DestroyRenderer(display->renderer);
	SDL_DestroyWindow(display->window);
	//Quit SDL subsystems
	SDL_Quit(); 

	free(display);
}

