#include <stdio.h>
#include <stdlib.h>
#include <SDL/SDL.h>
#include <math.h>

#define SCREEN_WIDTH 640
#define SCREEN_HEIGHT 480
#define PI 3.1415927

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
    vector2d start,plane;
    vector2d rotation;
}camera;

typedef struct{
    size_t pitch;
    size_t bpp;
    size_t w,h;
    void* pixels;
}gpu_surface;


#define ROTATION_ANGLE 0.1

void cameraEvalStart(camera* c){
    c->start.x=c->position.x+c->direction.x-c->plane.x;
    c->start.y=c->position.y+c->direction.y-c->plane.y;
}

camera initCamera(double x,double y,double fov){
    camera c;
    c.position.x=x;
    c.position.y=y;
    c.direction.x=0;
    c.direction.y=1;
    c.plane.x=tan(fov/2);
    c.plane.y=0;

    cameraEvalStart(&c);

    c.rotation.x=cos(ROTATION_ANGLE);
    c.rotation.y=sin(ROTATION_ANGLE);
    return c;
}

void cameraLeftRotation(camera* c){
    c->direction.x=c->rotation.x*c->direction.x-c->rotation.y*c->direction.y;
    c->direction.y=c->rotation.y*c->direction.x+c->rotation.x*c->direction.y;

    c->plane.x=c->rotation.x*c->plane.x-c->rotation.y*c->plane.y;
    c->plane.y=c->rotation.y*c->plane.x+c->rotation.x*c->plane.y;
    cameraEvalStart(c);
}

void cameraRightRotation(camera* c){
    c->direction.x=c->rotation.x*c->direction.x+c->rotation.y*c->direction.y;
    c->direction.y=c->rotation.x*c->direction.y-c->rotation.y*c->direction.x;

    c->plane.x=c->rotation.x*c->plane.x+c->rotation.y*c->plane.y;
    c->plane.y=c->rotation.x*c->plane.y-c->rotation.y*c->plane.x;
    cameraEvalStart(c);
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
    int ratio;
    int ysrc;

    if (dst_height<=0) dst_height=1;

    ystart=(dst->h>>1)-(dst_height>>1);
    yend=(dst->h>>1)+(dst_height>>1);
    if (ystart<0){
        ratio=-ystart;
        ystart=0;
        yend=dst->h;
    }else{
        ratio=0;
    }

    for (;ystart<=yend;ystart++,ratio++){
        ysrc=((double)ratio/dst_height)*src->h;
        //printf("ysrc=%d ydst=%d ratio=%d dst_height=%d\n",ysrc,ystart,ratio,dst_height);
        *(Uint32*)(dst->pixels+ystart*dst->pitch+dst_column*dst->bpp)=*(Uint32*)((src->pixels+ysrc*src->pitch+src_column*src->bpp));
    }
}

void myKernel(camera* c,gpu_surface* screen,gpu_surface* wall,int i){

    SDL_Rect source,dest;
    
    double xdirection,ydirection,xdiff,ydiff,xposition1,yposition1,xposition2,yposition2,xsign,ysign;
    double xdirection1,xdirection2,ydirection1,ydirection2,dist1,dist2,disttotal1,disttotal2;

    xposition1=c->position.x;
    yposition1=c->position.y;

    xdirection=c->start.x+c->plane.x*2*((double)i/screen->w)-xposition1;
    ydirection=c->start.y+c->plane.y*2*((double)i/screen->w)-yposition1;
    
    if (xdirection>=0){
        xdiff=(int)xposition1+1-xposition1;
        xsign=0.000001;
    }
    else{
        xdiff=xposition1-(int)xposition1;
        xsign=-0.000001;
    }

    if (ydirection>=0){
        ydiff=(int)yposition1+1-yposition1;
        ysign=0.000001;
    }
    else{
        ydiff=yposition1-(int)yposition1;
        ysign=-0.000001;
    }
    /*
    if (i==241){
        printf("xdirection=%lf\n",xdirection);
        printf("ydirection=%lf\n",ydirection);
        printf("xdirection2=%lf\n",xdirection2);
        printf("ydirection2=%lf\n",ydirection2);
        printf("xposition1=%lf\n",xposition1);
        printf("yposition1=%lf\n",yposition1);
        printf("x=%d y=%d\n",(int)(yposition1+ysign),(int)(xposition1+xsign));  
        printf("x=%d y=%d\n",(int)(yposition2+ysign),(int)(xposition2+xsign));  
        printf("dist1=%lf\n",dist1);
        printf("dist2=%lf\n",dist2);
        printf("xdiff=%lf\n",xdiff);
        printf("ydiff=%lf\n",ydiff);
    }
    */
    xdirection1=xdirection/fabs(xdirection);
    ydirection1=ydirection/fabs(xdirection);
    dist1=(1+ydirection1*ydirection1);

    xdirection2=xdirection/fabs(ydirection);
    ydirection2=ydirection/fabs(ydirection);
    dist2=(1+xdirection2*xdirection2);

    xposition2=xposition1+ydiff*xdirection2;
    yposition2=yposition1+ydiff*ydirection2;

    xposition1+=xdiff*xdirection1;
    yposition1+=xdiff*ydirection1;

    disttotal1=dist1*xdiff;
    disttotal2=dist2*ydiff;

    int height,column;
            
    
            

    while (1){
        
        if (disttotal1<disttotal2){
            if (map[(int)(yposition1+ysign)][(int)(xposition1+xsign)]){
                height=(double)300/disttotal1;
                column=(yposition1-(int)yposition1)*wall->w;
                break;
            }
            xposition1+=xdirection1;
            yposition1+=ydirection1;
            disttotal1+=dist1;
        }else{
            if (map[(int)(yposition2+ysign)][(int)(xposition2+xsign)]){
                height=(double)300/disttotal2;
                column=(xposition2-(int)xposition2)*wall->w;
                break;
            }
            xposition2+=xdirection2;
            yposition2+=ydirection2;
            disttotal2+=dist2;
        }
    }


    //printf("i=%d height=%d column=%d\n",i,height,column);
    gpuSurfaceCopyColumn(wall,screen,column,i,height);



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
    
    camera cam=initCamera(1.2,1.2,PI/2);
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
                        cam.position.x-=0.05*cam.direction.x;
                        cam.position.y-=0.05*cam.direction.y;
                    }
                    
                    break;
                case SDLK_RIGHT:
                    cameraRightRotation(&cam);
                    break;
                case SDLK_LEFT:
                    cameraLeftRotation(&cam);
                    break;
            }

        }
        printf("x=%lf y=%lf\n",cam.position.x,cam.position.y);
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