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

//bpp=4
typedef struct{
    size_t pitch;
    size_t w,h;
    void* pixels;
}gpu_surface;


#define ROTATION_ANGLE 0.05
#define CAMERA_SPEED 0.05

void evalCameraAngle(camera* c){
    c->direction.x=cos(c->angle);
    c->direction.y=sin(c->angle);
    c->plane.x=c->direction.y*CAMERA_FOV;
    c->plane.y=-c->direction.x*CAMERA_FOV;
}

gpu_surface* newGPUSurface(SDL_Surface* s){
    gpu_surface* gs=malloc(sizeof(gpu_surface));
    gs->pitch=s->pitch;
    gs->w=s->w;
    gs->h=s->h;
    gs->pixels=s->pixels;
}

 //use SDL1 instead of SDL2 beacause we want to do all compute on CPU and not use GPU at all


Uint32* gpuSurfaceGetPixel(gpu_surface* gs,Uint32 x,Uint32 y){
    return (Uint32*)(gs->pixels+y*gs->pitch+(x<<2));
}

void gpuSurfaceCopyColumn(gpu_surface* src,gpu_surface* dst,int src_column,int dst_column,int dst_height){
    int ystart,yend;
    int shift;

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

    double step=((double)1/dst_height)*src->h;
    double ysrc=shift*step;

    for (;ystart<=yend;ystart++,ysrc+=step){
        //printf("ysrc=%d ydst=%d ratio=%d dst_height=%d\n",ysrc,ystart,ratio,dst_height);
        *gpuSurfaceGetPixel(dst,dst_column,ystart)=*gpuSurfaceGetPixel(src,src_column,ysrc);
    }
}


void myKernel2(camera* c,gpu_surface* screen,gpu_surface* ground,int height,double distance,double rayDirX,double rayDirY,int x,int i){
    int p=height>>1;
    double w=p*distance;

    double d=w/((double)(p+i));
    double floorX=c->position.x+rayDirX*d;
    double floorY=c->position.y+rayDirY*d;

    int textX=(int)(ground->w*(floorX-(int)floorX)) & (ground->w-1);
    int textY=(int)(ground->h*(floorY-(int)floorY)) & (ground->h-1);

    *gpuSurfaceGetPixel(screen,x,(screen->h>>1)+p+i)=*gpuSurfaceGetPixel(ground,textX,textY);
}


void myKernel(camera* c,gpu_surface* screen,gpu_surface* wall,gpu_surface* ground,int i){
    double posX=c->position.x;
    double posY=c->position.y;

    double cameraX=((double)(2*i))/screen->w-1;
    double rayDirX=c->direction.x+c->plane.x*cameraX;
    double rayDirY=c->direction.y+c->plane.y*cameraX;

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
        if (sideDistX < sideDistY){
            sideDistX += deltaDistX;
            mapX += stepX;
            side = 0;
        }else{
            sideDistY += deltaDistY;
            mapY += stepY;
            side = 1;
        } 
    }while(!map[mapY][mapX]); 

    if(side == 0) perpWallDist = (sideDistX - deltaDistX);
    else          perpWallDist = (sideDistY - deltaDistY);
    
    int lineHeight = (double)500/ perpWallDist;
    //printf("i=%d perWalldist=%lf\n",i,perpWallDist);

    double wallX; //where exactly the wall was hit
    if (side == 0) wallX = posY + perpWallDist * rayDirY;
    else           wallX = posX + perpWallDist * rayDirX;
    wallX -= floor((wallX));

    gpuSurfaceCopyColumn(wall,screen,wallX*wall->w,i,lineHeight);

    int n=(screen->h>>1)-(lineHeight>>1);
    for (int j=0;j<n;j++){
        myKernel2(c,screen,ground,lineHeight,perpWallDist,rayDirX,rayDirY,i,j);
    }
      
}

int main(){
    SDL_Event event;
    SDL_Surface* temp;
    SDL_Init(SDL_INIT_VIDEO);
    SDL_Surface* screen = SDL_SetVideoMode(SCREEN_WIDTH, SCREEN_HEIGHT, 0, SDL_HWPALETTE);

    if (screen->format->BytesPerPixel==4){
        SDL_WM_SetCaption("test", NULL);

        temp=SDL_LoadBMP("brick.bmp");
        SDL_Surface* wall=SDL_ConvertSurface(temp,screen->format,0);
        SDL_FreeSurface(temp);

        temp=SDL_LoadBMP("stone.bmp");
        SDL_Surface* ground=SDL_ConvertSurface(temp,screen->format,0);
        SDL_FreeSurface(temp);

        gpu_surface* gpu_screen=newGPUSurface(screen);
        gpu_surface* gpu_wall=newGPUSurface(wall);
        gpu_surface* gpu_ground=newGPUSurface(ground);

        camera cam={
            .direction=(vector2d){0,1},
            .position=(vector2d){1.2,1.2},
            .plane=(vector2d){CAMERA_FOV,0},
            .angle=0
        };

        double nextx,nexty;
        
        
        while (1){
            for (int i=0;i<SCREEN_WIDTH;i++){
                myKernel(&cam,gpu_screen,gpu_wall,gpu_ground,i);
            }
        

            SDL_PollEvent(&event);
            
            if (event.type==SDL_QUIT) break;

            if (event.type==SDL_KEYDOWN){
                switch (event.key.keysym.sym){
                    case SDLK_RIGHT:
                        cam.angle-=ROTATION_ANGLE;
                        evalCameraAngle(&cam);
                        break;
                    case SDLK_LEFT:
                        cam.angle+=ROTATION_ANGLE;
                        evalCameraAngle(&cam);
                        break;

                    case SDLK_UP:
                        nextx=cam.position.x+CAMERA_SPEED*cam.direction.x;
                        nexty=cam.position.y+CAMERA_SPEED*cam.direction.y;
                        if (!map[(int)nexty][(int)nextx]){
                            cam.position.x=nextx;
                            cam.position.y=nexty;
                        }
                            

                        break;
                    case SDLK_DOWN:
                        nextx=cam.position.x-CAMERA_SPEED*cam.direction.x;
                        nexty=cam.position.y-CAMERA_SPEED*cam.direction.y;
                        if (!map[(int)nexty][(int)nextx]){
                            cam.position.x=nextx;
                            cam.position.y=nexty;
                        }
                        
                        break;
                    
                }

            }
            //printf("x=%lf y=%lf dx=%lf dy=%lf\n",cam.position.x,cam.position.y,cam.direction.x,cam.direction.y);
            SDL_Delay(10);
            SDL_Flip(screen);
            SDL_FillRect(screen,NULL,0x00000000);
        }

        free(gpu_screen);
        free(gpu_wall);

        SDL_FreeSurface(wall);
    }
    
    SDL_FreeSurface(screen);
    SDL_Quit();

    return EXIT_SUCCESS;
}