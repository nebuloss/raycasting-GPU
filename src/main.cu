/*
Merci de lire la version CPU avant, afin de comprendre le programme initial
Cette version est plus adapté pour des GPU anciens ayant peu de puissance 
ainsi que pour les iGPU utilisant une partie de la RAM comme VRAM.
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
__device__ void* gpuSurfaceGetPixel(gpu_surface* gs,Uint32 x,Uint32 y){
    return &((char*)gs->pixels)[y*gs->pitch+x*gs->bpp];
}

/*
Copie une colonne de pixel d'une gpu_surface source vers une autre colonne sur la gpu_surface destination. 
Cette fonction donne la possibilité d'étirer ou rétrécir la colonne sur la gpu_surface de destination.
La fonction est utilisés pour le tracé des murs.
*/
__device__ void gpuSurfaceCopyColumn(gpu_surface* src,gpu_surface* dst,int src_column,int dst_column,int dst_height){
    int ystart,yend;
    int shift;
    int bpp;

    bpp=src->bpp;
    if (bpp!=dst->bpp) return;
    if (dst_height<0) dst_height=0;

    ystart=(dst->h>>1)-(dst_height>>1);
    yend=ystart+dst_height;
    if (ystart<=0){
        shift=-ystart;
        ystart=0;
        yend=dst->h;
    }else{
        shift=0;
    }

    float step=((float)1/dst_height)*src->h;
    float ysrc=shift*step;

    for (;ystart<yend;ystart++,ysrc+=step){
        memcpy(gpuSurfaceGetPixel(dst,dst_column,ystart),gpuSurfaceGetPixel(src,src_column,ysrc),bpp);
    }
}

/*Sur la version GPU c'est un kernel cuda.
Cette fonction est responsable su tracé des murs sur l'écran.
explications sur ce site https://lodev.org/cgtutor/raycasting.html
*/
__global__ void wallCasting(camera* c,gpu_surface* screen,gpu_surface* wall,int* gpu_map){
    __shared__ int shared_map[MAP_WIDTH*MAP_HEIGHT];

    if (threadIdx.x<MAP_WIDTH*MAP_HEIGHT){
        shared_map[threadIdx.x]=gpu_map[threadIdx.x];
    }
    
    __syncthreads();

    int i = threadIdx.x+blockIdx.x*blockDim.x; // on récupère le i en fonction du numéro de thread et de block
    
    float posX=c->position.x;
    float posY=c->position.y;

    //pour s'adapter au matériel on calcule si necessaire plusieurs rayon pour 1 thread.
    while (i<screen->w){
        float cameraX=((float)(i<<1))/screen->w;
        float rayDirX=c->leftRayDir.x+c->plane.x*cameraX;
        float rayDirY=c->leftRayDir.y+c->plane.y*cameraX;

        float rayDirX2=rayDirX*rayDirX;
        float rayDirY2=rayDirY*rayDirY;

        float rayDist=sqrtf(rayDirX2+rayDirY2);

        float deltaDistX = rayDist/fabsf(rayDirX);
        float deltaDistY = rayDist/fabsf(rayDirY);

        int stepX,stepY;
        float sideDistX,sideDistY;
        int side; //was a NS or a EW wall hit?

        int mapX=posX;
        int mapY=posY;

        float perpWallDist;
        float wallX; //where exactly the wall was hit

        if (rayDirX < 0){
            stepX = -1;
            sideDistX = (posX - (float)mapX) * deltaDistX;
        }else{
            stepX = 1;
            sideDistX = ((float)mapX + 1.0 - posX) * deltaDistX;
        }if (rayDirY < 0){
            stepY = -1;
            sideDistY = (posY - (float)mapY) * deltaDistY;
        }else{
            stepY = 1;
            sideDistY = ((float)mapY + 1.0 - posY) * deltaDistY;
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
            wallX = posY + perpWallDist * rayDirY;
        }else{
            perpWallDist = sideDistY-deltaDistY;
            wallX = posX + perpWallDist * rayDirX;
        }
        
        int lineHeight = (float)screen->h/ perpWallDist*rayDist;
        wallX -= floor(wallX);

        gpuSurfaceCopyColumn(wall,screen,wallX*wall->w,i,lineHeight);
        i+=blockDim.x*gridDim.x;
    }    
}

/*Sur la version GPU c'est un kernel cuda.
Cette fonction est responsable su tracé du sol et du plafond l'écran.
explications sur ce site https://lodev.org/cgtutor/raycasting2.html.
Le tracé du sol ne dépend pas de la map. Il est dessiné sur tout l'écran puis recouvert du mur.
Contrairement au mur qui sont tracés colonne par colonne, le sol est tracé pixels par pixels dans le GPU.
*/
__global__ void floorCasting(camera* c,gpu_surface* screen,gpu_surface* floor,gpu_surface* ceil){
    float rowDistance,floorX,floorY;
    int tx,ty,x,y;

    int i=threadIdx.x+blockIdx.x*blockDim.x; //i varie de 0 au nombre de pixels de la moitié de la fenêtre.
    int bpp=screen->bpp;

    int screenWidth=screen->w;
    int halfScreenHeight=screen->h>>1;
    int totalPixels=halfScreenHeight*screenWidth;

    float ratioPlaneX=(c->plane.x*2.0F) / screenWidth; //pour éviter de recalculer la valeur à chaque itération on l'a séparé dans une autre variable
    float ratioPlaneY=(c->plane.y*2.0F) / screenWidth;

    int floorWidth=floor->w;
    int floorHeight=floor->h;

    while (i<totalPixels){
        y=i/screenWidth; //x associé à i
        x=i-y*screenWidth; //y associé à i
        rowDistance = (float)halfScreenHeight / y;

        floorX = c->position.x + rowDistance * (c->leftRayDir.x+ratioPlaneX*(float)x);
        floorY = c->position.y + rowDistance * (c->leftRayDir.y+ratioPlaneY*(float)x);

        tx = (int)(floorWidth * (floorX-(int)floorX)) & (floorWidth - 1);
        ty = (int)(floorHeight * (floorY-(int)floorY)) & (floorHeight - 1);

        memcpy(gpuSurfaceGetPixel(screen,x,y+halfScreenHeight),gpuSurfaceGetPixel(floor,tx,ty),bpp);
        memcpy(gpuSurfaceGetPixel(screen,x,halfScreenHeight-y-1),gpuSurfaceGetPixel(ceil,tx,ty),bpp);

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
        floorCasting<<<block_numbers,1024>>>(gpu_camera,gpu_screen,gpu_floor,gpu_ceil);
        cudaDeviceSynchronize(); //on attend que le sol ait fini d'être dessinné pour ne pas qu'il recouvre les murs
        wallCasting<<<2,1024>>>(gpu_camera,gpu_screen,gpu_wall,gpu_map);
        cudaDeviceSynchronize(); //on attend que le mur ait fini d'être rendu
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

        nextX=cam.position.x+stepX*GPU_CAMERA_SPEED;
        nextY=cam.position.y+stepY*GPU_CAMERA_SPEED;
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