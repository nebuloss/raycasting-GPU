#include <stdio.h>
#include <stdlib.h>
#include <SDL/SDL.h>
#include <math.h>

#define SCREEN_WIDTH 640
#define SCREEN_HEIGHT 480
#define PI 3.1415927
#define CAMERA_FOV 0.66

int map[20][20]={
    {1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1},
    {1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1},
    {1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1},
    {1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1},
    {1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1},
    {1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1},
    {1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1},
    {1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1},
    {1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1},
    {1,0,1,1,1,1,1,0,1,0,0,0,0,0,0,0,0,0,0,1},
    {1,0,1,0,0,0,1,0,1,0,1,0,1,1,1,1,1,1,0,1},
    {1,0,0,0,1,0,1,0,1,0,1,0,1,0,0,0,0,1,0,1},
    {1,1,1,1,1,0,1,0,1,0,1,0,1,0,0,0,0,1,0,1},
    {1,0,0,0,0,0,1,0,1,0,1,0,1,0,0,0,0,1,0,1},
    {1,0,1,1,1,1,1,0,1,0,1,0,1,0,0,0,0,1,0,1},
    {1,0,1,0,0,0,0,0,1,0,1,0,1,0,0,0,0,1,0,1},
    {1,0,1,1,1,1,1,1,1,0,1,0,1,0,0,0,0,1,0,1},
    {1,0,1,0,0,0,1,0,0,0,1,0,1,0,1,1,1,1,0,1},
    {1,0,0,0,1,0,0,0,1,0,1,0,0,0,1,1,1,0,0,1},
    {1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1},
};

typedef struct{
    double x,y;
}vector2d;

typedef struct{
    vector2d position;
    vector2d direction;
    vector2d plane;
    double angle;
}camera;

typedef struct{
    size_t pitch;
    size_t bpp;
    size_t w,h;
    void* pixels;
}gpu_surface;


#define ROTATION_ANGLE 0.1

void evalCameraAngle(camera* c){
    c->direction.x=cos(c->angle);
    c->direction.y=sin(c->angle);
    c->plane.x=c->direction.y*CAMERA_FOV;
    c->plane.y=-c->direction.x*CAMERA_FOV;
}

gpu_surface* newGPUSurface(SDL_Surface* s){
    gpu_surface* gs=malloc(sizeof(gpu_surface));
    gs->bpp=s->format->BytesPerPixel;
    gs->pitch=s->pitch;
    gs->w=s->w;
    gs->h=s->h;
    gs->pixels=s->pixels;
}

 //use SDL1 instead of SDL2 beacause we want to do all compute on CPU and not use GPU at all

void gpuSurfaceCopyColumn(gpu_surface* src,gpu_surface* dst,int src_column,int dst_column,int dst_height){
    int ystart,yend;
    int shift;
    int ysrc;

    if (dst_height<=0) dst_height=1;

    ystart=(dst->h>>1)-(dst_height>>1);
    yend=(dst->h>>1)+(dst_height>>1);
    if (ystart<0){
        shift=-ystart;
        ystart=0;
        yend=dst->h;
    }else{
        shift=0;
    }

    for (;ystart<=yend;ystart++,shift++){
        ysrc=((double)shift/dst_height)*src->h;
        //printf("ysrc=%d ydst=%d ratio=%d dst_height=%d\n",ysrc,ystart,ratio,dst_height);
        *(Uint32*)(dst->pixels+ystart*dst->pitch+dst_column*dst->bpp)=*(Uint32*)((src->pixels+ysrc*src->pitch+src_column*src->bpp));
    }
}

void myKernel(camera* c,gpu_surface* screen,gpu_surface* wall,int i){
    double posX=c->position.x;
    double posY=c->position.y;

    double cameraX=((double)(2*i))/screen->w-1;
    double rayDirX=posX+c->plane.x*cameraX;
    double rayDirY=posY+c->plane.y*cameraX;

    double deltaDistX = sqrt(1 + (rayDirY * rayDirY) / (rayDirX * rayDirX));
    double deltaDistY = sqrt(1 + (rayDirX * rayDirX) / (rayDirY * rayDirY));

    int stepX,stepY;
    double sideDistX,sideDistY;

    int side; //was a NS or a EW wall hit?

    int mapX=posX;
    int mapY=posY;

    double perpWallDist;

    //calculate step and initial sideDist
      if (rayDirX < 0){
        stepX = -1;
        sideDistX = (posX - (double)mapX) * deltaDistX;
      }else{
        stepX = 1;
        sideDistX = ((double)mapX + 1.0 - posX) * deltaDistX;
      }if (rayDirY < 0){
        stepY = -1;
        sideDistY = (posY - (double)mapY) * deltaDistY;
      }else{
        stepY = 1;
        sideDistY = ((double)mapY + 1.0 - posY) * deltaDistY;
      }

      //perform DDA
      do{
        //jump to next map square, either in x-direction, or in y-direction
        if (sideDistX < sideDistY)
        {
          sideDistX += deltaDistX;
          mapX += stepX;
          side = 0;
        }
        else
        {
          sideDistY += deltaDistY;
          mapY += stepY;
          side = 1;
        }
        
      }while(!map[mapX][mapY]); 

      if(side == 0) perpWallDist = (sideDistX - deltaDistX);
      else          perpWallDist = (sideDistY - deltaDistY);
     
      int lineHeight = (double)2000 / perpWallDist;
      //printf("i=%d perWalldist=%lf\n",i,perpWallDist);

      double wallX; //where exactly the wall was hit
      if (side == 0) wallX = posY + perpWallDist * rayDirY;
      else           wallX = posX + perpWallDist * rayDirX;
      wallX -= floor((wallX));

      gpuSurfaceCopyColumn(wall,screen,wallX*wall->w,i,lineHeight);
}

int main(){
    SDL_Event event;
    SDL_Init(SDL_INIT_VIDEO);
    SDL_Surface* screen = SDL_SetVideoMode(SCREEN_WIDTH, SCREEN_HEIGHT, 0, SDL_HWPALETTE);
    SDL_WM_SetCaption("test", NULL);

    SDL_Surface* temp=SDL_LoadBMP("brick.bmp");
    SDL_Surface* wall=SDL_ConvertSurface(temp,screen->format,0);
    SDL_FreeSurface(temp);

    gpu_surface* gpu_screen=newGPUSurface(screen);
    gpu_surface* gpu_wall=newGPUSurface(wall);

    camera cam={
        .direction=(vector2d){0,1},
        .position=(vector2d){1.2,1.2},
        .plane=(vector2d){CAMERA_FOV,0},
        .angle=0
    };
    
    /*
    for (int i=0;i<gpu_screen->w;i++){
        gpuSurfaceCopyColumn(gpu_wall,gpu_screen,1,i,i);
    }
    */
     
    
    while (1){
        for (int i=0;i<SCREEN_WIDTH;i++){
            myKernel(&cam,gpu_screen,gpu_wall,i);
        }
     

        SDL_PollEvent(&event);
        
        if (event.type==SDL_QUIT) break;

        if (event.type==SDL_KEYDOWN){
            switch (event.key.keysym.sym){
                case SDLK_UP:
                    if (!map[(int)(cam.position.y+0.1*cam.direction.y)][(int)(cam.position.x+0.1*cam.direction.x)]){
                        cam.position.x+=0.01*cam.direction.x;
                        cam.position.y+=0.01*cam.direction.y;
                    }
                        

                    break;
                case SDLK_DOWN:
                    if (!map[(int)(cam.position.y-0.1*cam.direction.y)][(int)(cam.position.x-0.1*cam.direction.x)]){
                        cam.position.x-=0.1*cam.direction.x;
                        cam.position.y-=0.1*cam.direction.y;
                    }
                    
                    break;
                case SDLK_RIGHT:
                    cam.angle-=0.05;
                    evalCameraAngle(&cam);
                    break;
                case SDLK_LEFT:
                    cam.angle+=0.05;
                    evalCameraAngle(&cam);
                    break;
            }

        }
        printf("x=%lf y=%lf dx=%lf dy=%lf\n",cam.position.x,cam.position.y,cam.direction.x,cam.direction.y);
        SDL_Delay(10);
        SDL_Flip(screen);
        SDL_FillRect(screen,NULL,0x00000000);
    }

    free(gpu_screen);
    free(gpu_wall);

    SDL_FreeSurface(wall);
    SDL_FreeSurface(screen);
    SDL_Quit();

    return EXIT_SUCCESS;
}