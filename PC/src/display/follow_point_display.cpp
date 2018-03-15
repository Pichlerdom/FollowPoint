

#include "followPoint.h"
#include "follow_point_display.h"
#include "webcam_interface.h"



void* main_display_loop(void *arg){
	
	Thread_Args *args = (Thread_Args *) arg;	
	Display *display = init_display();
	SDL_Event e;

	TTF_Init();
	TTF_Font *font = TTF_OpenFont("RobotoMono-Medium.ttf", 12);
	SDL_Texture *texture;
	SDL_Rect textrect;
	char *tempstr = (char*) calloc(256,sizeof(char));
	

	bool mousekeydown = false;
	bool upkeydown = false;
	bool downkeydown = false;

	uint32_t currTime = SDL_GetTicks();
	uint32_t frameTime = 0u;
	uint32_t frameTimeDraw = 1u;
	
	int mousex = 0;
	int mousey = 0;

	while(args->control.display){
		currTime = SDL_GetTicks();		
		
		SDL_GetMouseState(&mousex, &mousey);
		//event handling loop
		while(SDL_PollEvent(&e) != 0){

			switch(e.type){
			
				//close application
				case SDL_QUIT:
					args->control.display = false;
					args->control.capture = false;
					args->control.detection = false;
				break;
				case SDL_MOUSEBUTTONUP:
					pthread_mutex_lock(&(args->comp_mtx));
					for(int dy = 0; dy < COMP_HEIGHT; dy++){
						for(int dx = 0; dx < COMP_WIDTH; dx++){
							if(	mousex + dx < args->image_buffer.width &&
								mousey + dy < args->image_buffer.height) {
								
								int pixIdx = (WEBCAM_FRAME_WIDTH * (mousey + dy) + (mousex + dx)) * 3 ;
								int compIdx = (COMP_WIDTH * dy + dx) * 3;
								args->comp[compIdx] = args->image_buffer.read[pixIdx];
								args->comp[compIdx + 1] = args->image_buffer.read[pixIdx + 1];
								args->comp[compIdx + 2] = args->image_buffer.read[pixIdx + 2];
							}
						}
					}
					pthread_mutex_unlock(&(args->comp_mtx));
					mousekeydown = false;
				break;
				case SDL_MOUSEBUTTONDOWN:
					mousekeydown = true;
				break;
				case SDL_KEYDOWN:
					switch( e.key.keysym.sym ){
						case SDLK_UP:
								upkeydown = true;
							break;
						case SDLK_DOWN:
								downkeydown = true;
							break;
						default:
							break;
					}
				break;

				case SDL_KEYUP:

					switch( e.key.keysym.sym ){
						case SDLK_UP:
								upkeydown = false;
							break;
						case SDLK_DOWN:
								downkeydown = false;
							break;
						default:
							break;
					}
				break;
			}
		}

		
		//Clear screen
		SDL_SetRenderDrawColor( display->renderer, 0x00, 0x00, 0x00, 0xFF );
		SDL_RenderClear( display->renderer );

		/************ DRAW IMAGES **************/
		if(args->image_buffer.read != NULL){
			draw_img(display, args->image_buffer.read,
					 args->image_buffer.width, args->image_buffer.height,
					 0, 0);
			draw_img_grey(display, args->image_buffer.proc,
					 args->image_buffer.width ,
					 args->image_buffer.height,
					 args->image_buffer.width + (COMP_WIDTH/2), (COMP_HEIGHT/2));

		}
	

		if(args->comp != NULL){
			draw_img(display, args->comp,
					 COMP_HEIGHT, COMP_WIDTH,
					 0, args->image_buffer.height);
		}


		/************ KEY FUNCTIONS **********/
		if(mousekeydown){

			SDL_Rect rect;
			rect.x = mousex;
			rect.y = mousey;
			rect.h = COMP_HEIGHT;
			rect.w = COMP_WIDTH;

			/******************  DRAW SELECTION BOX ***************/
			SDL_SetRenderDrawColor( display->renderer, 0xFF, 0x00, 0xFF, 0xFF);
			SDL_RenderDrawRect(display->renderer, &rect);
		
			pthread_mutex_lock(&(args->laser.mtx));
			args->laser.dx = 0;
			args->laser.dy = 0;
			pthread_mutex_unlock(&(args->laser.mtx));
		}
		if(upkeydown){
			args->threshold+=0.1;
		}
		if(downkeydown){
			args->threshold-=0.1;
			if(args->threshold < 0){
				args->threshold = 0;
			}
		}



		/*********** DRAW LASER ***************/

		SDL_SetRenderDrawColor( display->renderer, 0xFF, 0x00, 0x00, 0xFF);
		draw_circle_at(display, 2, args->laser.x, args->laser.y);
		
		SDL_SetRenderDrawColor( display->renderer, 0x00, 0xFF, 0x00, 0xFF);
		draw_circle_at(display, 1, WEBCAM_FRAME_WIDTH/2, WEBCAM_FRAME_HEIGHT/2);
		
		/********** DRAW TEXT *****************/
		if(args->capturetime > 0){
			sprintf(tempstr,"FPS:%d", 1000/args->capturetime);
			get_text_and_rect(display->renderer,
							  10, 10,
							  tempstr,
			font, &texture, &textrect);
			SDL_RenderCopy(display->renderer, texture, NULL, &textrect);
		}
		if(args->calctime > 0){
			sprintf(tempstr,"FPS:%d", 1000/args->calctime);
			get_text_and_rect(	display->renderer,
								WEBCAM_FRAME_WIDTH + 10 + (COMP_WIDTH/2), (COMP_HEIGHT/2)+10,
								tempstr,
			font, &texture, &textrect);
			SDL_RenderCopy(display->renderer, texture, NULL, &textrect);
		}

		sprintf(tempstr,"Threshold:%.2f", args->threshold);
		get_text_and_rect(	display->renderer,
							COMP_WIDTH, WEBCAM_FRAME_HEIGHT,
							tempstr,
		font, &texture, &textrect);
		SDL_RenderCopy(display->renderer, texture, NULL, &textrect);
		
		sprintf(tempstr,"Laser(dx/dy):%d/%d", args->laser.dx,args->laser.dy);
		get_text_and_rect(	display->renderer,
							COMP_WIDTH + 150, WEBCAM_FRAME_HEIGHT,
							tempstr,
		font, &texture, &textrect);
		SDL_RenderCopy(display->renderer, texture, NULL, &textrect);
		
		sprintf(tempstr,"Pixel Count:%d", args->laser.pixcount);
		get_text_and_rect(	display->renderer,
							COMP_WIDTH + 350, WEBCAM_FRAME_HEIGHT,
							tempstr,
		font, &texture, &textrect);
		SDL_RenderCopy(display->renderer, texture, NULL, &textrect);
		
		
		//Update screen
		SDL_RenderPresent( display->renderer );
		

		//FPS stuff
		frameTime = SDL_GetTicks() - currTime;
		frameTimeDraw = frameTime;
		if(frameTime > MS_PER_FRAME){
			frameTime = MS_PER_FRAME;
		}
		SDL_Delay(MS_PER_FRAME-frameTime);

	}

	printf("Display thread exited!\n");
	close(display);
	return NULL;

}

void draw_circle_at(Display* display, 
					int radius, 
					int xpos, int ypos){

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

void draw_img(	Display* display, uint8_t *img,
				int width, int height,
				int xpos, int ypos){

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

void draw_img_grey(	Display* display, uint8_t *img,
					int width, int height,
					int xpos, int ypos){
	for(int y = 0;  y < height; y++){
		for(int x = 0; x < width; x++){

			int pixIdx = (width * y + x);
			uint8_t g = img[pixIdx];
			if(	x < WEBCAM_FRAME_WIDTH - COMP_WIDTH && 
				y < WEBCAM_FRAME_HEIGHT - COMP_HEIGHT){
			}
			if(	g < 1 ){	
				SDL_SetRenderDrawColor(display->renderer, 0x00, 0x00, 0x00, 0xff);
			}else{
				SDL_SetRenderDrawColor(display->renderer, 255 - g, g, 0x00, 0xff);
			}
			if(	x < WEBCAM_FRAME_WIDTH - COMP_WIDTH && 
				y < WEBCAM_FRAME_HEIGHT - COMP_HEIGHT){
				SDL_RenderDrawPoint(display->renderer, x + xpos, y + ypos);
			}
		}
	}
}

/*
- x, y: upper left corner.
- texture, rect: outputs.
*/
void get_text_and_rect(	SDL_Renderer *renderer, 
						int x, int y,
						char *text,
						TTF_Font *font, SDL_Texture **texture, SDL_Rect *rect) {
    int text_width;
    int text_height;
    SDL_Surface *surface;
    SDL_Color textColor = {255, 255, 255, 0};

    surface = TTF_RenderText_Solid(font, text, textColor);
    *texture = SDL_CreateTextureFromSurface(renderer, surface);
    text_width = surface->w;
    text_height = surface->h;
    SDL_FreeSurface(surface);
    rect->x = x;
    rect->y = y;
    rect->w = text_width;
    rect->h = text_height;
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
	display->renderer = SDL_CreateRenderer( display->window, -1, SDL_RENDERER_SOFTWARE );
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

