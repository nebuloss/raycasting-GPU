/*
Nouvelle implémention qui permet de corriger le ralentissement lorsque qu'on est proche d'un mur.
On utilise maintenant plus qu'un seul kernel pour tracer le sol et le mur.
Cette version est plus adapté pour les GPU récents
*/

#include <stdio.h>
#include <stdlib.h>
#include <SDL/SDL.h>
#include "main.h"

/*
Pour la version GPU, on créé une structure simplifié par rapport SDL_Surface pour permettre
de copier les éléments indispensable d'une surface dans le GPU.
*/
typedef struct{
    Uint32 pitch,bpp; //pitch=taille d'une ligne de pixels (en octet)
    int w,h;
    void* pixels; //pointer vers les pixels
}gpu_surface;

/*
Alloue une gpu_surface dans le GPU et copie les pixels de la surface dans le GPU.
Cette fonction retourne l'adresse de la gpu_surface dans le GPU.
*/
gpu_surface* allocGPUSurface(SDL_Surface* s){
    void* pixels;
    gpu_surface *dst,src;
    size_t lenght=s->pitch*s->h;

    cudaMalloc(&dst,sizeof(gpu_surface));
    cudaMalloc(&pixels,lenght);

    cudaMemcpy(pixels,s->pixels,lenght,cudaMemcpyHostToDevice);
    
    src=(gpu_surface){
        .pitch=s->pitch,
        .bpp=s->format->BytesPerPixel,
        .w=s->w,
        .h=s->h,
        .pixels=pixels
    };
    cudaMemcpy(dst,&src,sizeof(gpu_surface),cudaMemcpyHostToDevice);
    return dst;
}

//libère les ressources prises par la gpu_surface dans le GPU
void freeGPUSurface(gpu_surface* gs){
    gpu_surface dst;
    cudaMemcpy(&dst,gs,sizeof(gpu_surface),cudaMemcpyDeviceToHost);
    cudaFree(dst.pixels);
    cudaFree(gs);
}

/*
Cette fonction est utilisée pour rafraichir l'écran à chaque frame.
La fonction permet de copier les pixels d'une gpu_surface dans la VRAM vers une surface dans la RAM.
*/
void gpuSurfaceCopytoHost(gpu_surface* gs,SDL_Surface* s){
    gpu_surface dst;
    cudaMemcpy(&dst,gs,sizeof(gpu_surface),cudaMemcpyDeviceToHost);
    if (dst.w!=s->w || dst.h!=s->h) return; //on vérifie que les surfaces ont la même dimension
    cudaMemcpy(s->pixels,dst.pixels,dst.pitch*dst.h,cudaMemcpyDeviceToHost);
}

void evalCameraAngle(camera* c){
    c->direction.x=cos(c->angle);
    c->direction.y=sin(c->angle);
    c->plane.x=c->direction.y*CAMERA_FOV;
    c->plane.y=-c->direction.x*CAMERA_FOV;
    c->leftRayDir.x=c->direction.x-c->plane.x;
    c->leftRayDir.y=c->direction.y-c->plane.y;
}

//retourne un pointer vers le pixel correspondant dans une gpu_surface
__device__ inline void* gpuSurfaceGetPixel(gpu_surface* gs,Uint32 x,Uint32 y){
    return &((char*)gs->pixels)[y*gs->pitch+x*gs->bpp];
}

/*Sur la version GPU c'est un kernel cuda.
Cette fonction est responsable su tracé du sol et du plafond l'écran.
explications sur ce site https://lodev.org/cgtutor/raycasting2.html.
Le tracé du sol ne dépend pas de la map. Il est dessiné sur tout l'écran puis recouvert du mur.
Contrairement au mur qui sont tracés colonne par colonne, le sol est tracé pixels par pixels dans le GPU.
*/
__global__ void rayCasting(camera* c,gpu_surface* gs_screen,gpu_surface* gs_wall,gpu_surface* gs_floor,gpu_surface* gs_ceil,int* gpu_map){
    __shared__ int shared_map[MAP_WIDTH*MAP_HEIGHT];
    __shared__ camera shared_camera;
    __shared__ gpu_surface shared_gs_screen;
    __shared__ gpu_surface shared_gs_wall;
    __shared__ gpu_surface shared_gs_floor;
    __shared__ gpu_surface shared_gs_ceil;

    if (threadIdx.x<MAP_WIDTH*MAP_HEIGHT){
        shared_map[threadIdx.x]=gpu_map[threadIdx.x];
    }else if (threadIdx.x==MAP_WIDTH*MAP_HEIGHT){
        memcpy(&shared_gs_screen,gs_screen,sizeof(gpu_surface));
    }else if (threadIdx.x==MAP_WIDTH*MAP_HEIGHT+1){
        memcpy(&shared_gs_wall,gs_wall,sizeof(gpu_surface));
    }else if (threadIdx.x==MAP_WIDTH*MAP_HEIGHT+2){
        memcpy(&shared_gs_floor,gs_floor,sizeof(gpu_surface));
    }else if (threadIdx.x==MAP_WIDTH*MAP_HEIGHT+3){
        memcpy(&shared_gs_ceil,gs_ceil,sizeof(gpu_surface));
    }else if (threadIdx.x==MAP_WIDTH*MAP_HEIGHT+4){
        memcpy(&shared_camera,c,sizeof(camera));
    }
    __syncthreads();
    
    int tx,ty,x,y;
    float rowDistance,floorX,floorY;
    int stepX,stepY;
    float sideDistX,sideDistY;
    int side,ystart,xsrc,ysrc; //was a NS or a EW wall hit?

    int i=threadIdx.x+blockIdx.x*blockDim.x; //i varie de 0 au nombre de pixels de la moitié de la fenêtre.

    int halfScreenHeight=shared_gs_screen.h>>1;

    int totalPixels=halfScreenHeight*shared_gs_screen.w;

    while (i<totalPixels){
        y=i/shared_gs_screen.w; //x associé à i
        x=i-y*shared_gs_screen.w; //y associé à i

        float cameraX=((float)(x<<1))/shared_gs_screen.w;
        float rayDirX=shared_camera.leftRayDir.x+shared_camera.plane.x*cameraX;
        float rayDirY=shared_camera.leftRayDir.y+shared_camera.plane.y*cameraX;

        float rayDirX2=rayDirX*rayDirX;
        float rayDirY2=rayDirY*rayDirY;

        float rayDist=sqrtf(rayDirX2+rayDirY2);

        float deltaDistX = rayDist/fabsf(rayDirX);
        float deltaDistY = rayDist/fabsf(rayDirY);

        int mapX=shared_camera.position.x;
        int mapY=shared_camera.position.y;

        float perpWallDist;
        float wallX; //where exactly the wall was hit

        if (rayDirX < 0){
            stepX = -1;
            sideDistX = (shared_camera.position.x - (float)mapX) * deltaDistX;
        }else{
            stepX = 1;
            sideDistX = ((float)mapX + 1.0 - shared_camera.position.x) * deltaDistX;
        }if (rayDirY < 0){
            stepY = -1;
            sideDistY = (shared_camera.position.y - (float)mapY) * deltaDistY;
        }else{
            stepY = 1;
            sideDistY = ((float)mapY + 1.0 - shared_camera.position.y) * deltaDistY;
        }

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
        }while(!shared_map[mapY*MAP_WIDTH+mapX]); // les tableaux statiques 2d sont remplacés par des tableau 1d à la compilation.

        if(!side){
            perpWallDist = sideDistX-deltaDistX;
            wallX = shared_camera.position.y + perpWallDist * rayDirY;
        }else{
            perpWallDist = sideDistY-deltaDistY;
            wallX = shared_camera.position.x + perpWallDist * rayDirX;
        }
        
        int lineHeight = (float)shared_gs_screen.h/ perpWallDist*rayDist;
        wallX -= floor(wallX);
        ystart=halfScreenHeight-(lineHeight>>1);

        if (y<ystart){
            rowDistance = (float)halfScreenHeight / (halfScreenHeight-y-1);

            floorX = shared_camera.position.x + rowDistance * rayDirX;
            floorY = shared_camera.position.y + rowDistance * rayDirY;
            floorX-=floor(floorX);
            floorY-=floor(floorY);

            tx = shared_gs_floor.w*floorX;
            ty = shared_gs_floor.h*floorY;

            memcpy(gpuSurfaceGetPixel(&shared_gs_screen,x,y),gpuSurfaceGetPixel(&shared_gs_ceil,tx,ty),shared_gs_screen.bpp);
            memcpy(gpuSurfaceGetPixel(&shared_gs_screen,x,shared_gs_screen.h-y-1),gpuSurfaceGetPixel(&shared_gs_floor,tx,ty),shared_gs_screen.bpp);
    
        }else{
            xsrc=wallX*shared_gs_wall.w;
            ysrc=((float)(y-ystart))/lineHeight*shared_gs_wall.h;
            memcpy(gpuSurfaceGetPixel(&shared_gs_screen,x,y),gpuSurfaceGetPixel(&shared_gs_wall,xsrc,ysrc),shared_gs_screen.bpp);
            memcpy(gpuSurfaceGetPixel(&shared_gs_screen,x,shared_gs_screen.h-y-1),gpuSurfaceGetPixel(&shared_gs_wall,xsrc,shared_gs_wall.h-ysrc-1),shared_gs_screen.bpp);
        }

        i+=blockDim.x*gridDim.x;
    }
}


SDL_Surface* loadBitMapFormat(char* filename,SDL_PixelFormat* fmt){
    SDL_Surface* temp=SDL_LoadBMP(filename);
    SDL_Surface* s=SDL_ConvertSurface(temp,fmt,0);
    SDL_FreeSurface(temp);
    return s;
}

int main(int argc,char* argv[]){
    int relX;
    Uint8* keystate;
    double nextX,nextY,stepX,stepY;
    camera cam,*gpu_camera;
    SDL_Event event;
    int* gpu_map;
    float elapsed_time;
    size_t block_numbers;

    //création des évènements cuda
    cudaEvent_t start,stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    if (argc==3){
        SCREEN_WIDTH=atoi(argv[1]);
        SCREEN_HEIGHT=atoi(argv[2]);
    }

    //initialisation de la SDL
    SDL_Init(SDL_INIT_VIDEO);
    SDL_Surface* screen = SDL_SetVideoMode(SCREEN_WIDTH, SCREEN_HEIGHT, 0, SDL_NOFRAME);
    SDL_ShowCursor(SDL_DISABLE);
    SDL_WM_GrabInput(SDL_GRAB_ON);

    block_numbers=(screen->pitch*screen->h>>11)+1; //nombre de blocs necessaires pour atteindre des bonnes perforamnces (=(nb_pixels écran/ 2048) +1)

    SDL_Surface* wall_surface=loadBitMapFormat("brick.bmp",screen->format);
    SDL_Surface* floor_surface=loadBitMapFormat("stone.bmp",screen->format);
    SDL_Surface* ceil_surface=loadBitMapFormat("wood.bmp",screen->format);

    SDL_WM_SetIcon(floor_surface,NULL); //set icon x)
    
    //en plus des surfaces dans la RAM, on alloue les gpu_surfaces dans le VRAM
    gpu_surface* gpu_screen=allocGPUSurface(screen);
    gpu_surface* gpu_wall=allocGPUSurface(wall_surface);
    gpu_surface* gpu_floor=allocGPUSurface(floor_surface);
    gpu_surface* gpu_ceil=allocGPUSurface(ceil_surface);

    //on alloue et copie la map dans la VRAM
    cudaMalloc(&gpu_map,sizeof(int)*MAP_WIDTH*MAP_HEIGHT);
    cudaMemcpy(gpu_map,map,sizeof(int)*MAP_WIDTH*MAP_HEIGHT,cudaMemcpyHostToDevice);

    cam.position=(vector2f){4,4};
    cam.angle=0;
    evalCameraAngle(&cam);

    //alloue la caméra dans le GPU
    cudaMalloc(&gpu_camera,sizeof(camera));
    
    while (1){

        cudaEventRecord(start); // évènement de début de la frame

        cudaMemcpy(gpu_camera,&cam,sizeof(camera),cudaMemcpyHostToDevice);
        rayCasting<<<block_numbers,1024>>>(gpu_camera,gpu_screen,gpu_wall,gpu_floor,gpu_ceil,gpu_map);
        cudaDeviceSynchronize(); //on attend que le sol ait fini d'être dessinné pour ne pas qu'il recouvre les murs
        gpuSurfaceCopytoHost(gpu_screen,screen);   //on copie vers la surface correspondant à l'écran le rendu
        cudaEventRecord(stop); //évènement de la fin de la frame
        
        
        cudaEventSynchronize(stop); //on attend que tout soit fini
        cudaEventElapsedTime(&elapsed_time,start,stop); //calul su temps écoulé
        //printf("%f\n",elapsed_time);
        
        SDL_PumpEvents();

        SDL_GetRelativeMouseState(&relX,NULL);
        cam.angle-=relX*ROTATION_ANGLE; //décrémente l'angle
        evalCameraAngle(&cam);

        stepX=0;
        stepY=0;
        
        keystate=SDL_GetKeyState(NULL);
        if (keystate[SDLK_ESCAPE]) break;
        if (keystate[SDLK_z]){
            stepX+=cam.direction.x;
            stepY+=cam.direction.y;
        }
        if (keystate[SDLK_s]){
            stepX-=cam.direction.x;
            stepY-=cam.direction.y;
        }
        if (keystate[SDLK_q]){
            stepX-=cam.direction.y;
            stepY+=cam.direction.x;
        }
        if (keystate[SDLK_d]){
            stepX+=cam.direction.y;
            stepY-=cam.direction.x;
        }

        nextX=cam.position.x+stepX*CAMERA_SPEED;
        nextY=cam.position.y+stepY*CAMERA_SPEED;
        if (!map[(int)nextY][(int)nextX]){
            cam.position.x=nextX;
            cam.position.y=nextY;
        }
        
        SDL_Flip(screen);  //met à jour l'écran
    }
    //on s'assure que les kernels cuda aient fini de s'éxécuter avant de quitter
    cudaDeviceSynchronize();
    cudaFree(gpu_map);
    cudaFree(gpu_camera);

    SDL_FreeSurface(wall_surface);
    SDL_FreeSurface(floor_surface);
    SDL_FreeSurface(ceil_surface);

    //libération des ressources gpu
    freeGPUSurface(gpu_screen);
    freeGPUSurface(gpu_floor);
    freeGPUSurface(gpu_ceil);
    freeGPUSurface(gpu_wall);
    
    SDL_FreeSurface(screen);
    SDL_Quit();

    return EXIT_SUCCESS;
}