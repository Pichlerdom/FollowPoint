
#ifndef _FOLLOW_POINT_DISPLAY_H_
#define _FOLLOW_POINT_DISPLAY_H_

#include <SDL2/SDL.h>
#include <SDL2/SDL_image.h>
#include <SDL2/SDL_ttf.h>

#define FPS 33
#define MS_PER_FRAME 1000/FPS

typedef struct{
	SDL_Renderer* renderer;
	SDL_Window* window;
}Display;


void* main_display_loop(void *imgCont);

void draw_circle_at(Display* display, int radius, int xpos, int ypos);

void draw_img(Display* display,uint8_t *img, int width, int height, int xpos, int ypos);

void draw_img_grey(Display* display,uint8_t *img, int width, int height, int xpos, int ypos);

void get_text_and_rect(SDL_Renderer *renderer, int x, int y, char *text,
        TTF_Font *font, SDL_Texture **texture, SDL_Rect *rect);

void yuv_to_rgb (uint8_t* yuv, uint8_t * rgb);

void close(Display* display);

Display* init_display();

#endif
