
#ifndef _FOLLOW_POINT_DISPLAY_H_
#define _FOLLOW_POINT_DISPLAY_H_



#define FPS 10
#define MS_PER_FRAME 1000/FPS

typedef struct{
	SDL_Renderer* renderer;
	SDL_Window* window;
}Display;


void* main_display_loop(void *imgCont);

void draw_circle_at(Display* display, int radius, int xpos, int ypos);

void draw_img(Display* display,uint8_t *img, int width, int height, int xpos, int ypos);

void draw_img_grey(Display* display,uint8_t *img, int width, int height, int xpos, int ypos);

void yuv_to_rgb (uint8_t* yuv, uint8_t * rgb);

void close(Display* display);

Display* init_display();

#endif
