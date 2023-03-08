#include <stdio.h>
#include <stdlib.h>
#include <SDL/SDL.h>

 //use SDL1 instead of SDL2 beacause we want to do all compute on CPU and not use GPU at all
int main(){

    SDL_Init(SDL_INIT_VIDEO);
    SDL_Rect r={0,0,200,200};
    SDL_Surface* screen = SDL_SetVideoMode(640, 480, 0, SDL_HWPALETTE);
    SDL_WM_SetCaption("test", NULL);
    SDL_FillRect(screen,&r,SDL_MapRGB(screen->format,255,0,0));
    SDL_Flip(screen);
    SDL_Delay(2000);
    SDL_Quit();

    return EXIT_SUCCESS;
}