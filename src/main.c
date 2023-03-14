#include <stdio.h> 
#include <stdlib.h> 
#include <SDL/SDL.h> //SDL 1.2 utilise le CPU tandis que SDL2 le GPU
#include <string.h> //memcpy
#include <math.h> //cos,sin
#include <time.h> //clock -> pour mesurer le temps

#define ROTATION_ANGLE 0.003 //vitesse de rotation
#define CAMERA_SPEED 0.05   //vitesse à laquelle on avance

int SCREEN_WIDTH=620; //largeur de la fenêtre par défaut
int SCREEN_HEIGHT=480; //hauteur de la fenêtre par défaut
const double CAMERA_FOV=0.66; //fov 

//description de la map (tableau 2d -> 0=vide 1=mur)
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

//vecteur 2 dimensions avec composante flottante
typedef struct{
    float x,y;
}vector2f;

//structure de la caméra
typedef struct{
    vector2f position; //vecteur position
    vector2f direction; //vecteur direction
    vector2f plane;  //vecteur plan (orthogonal à la direction)
    vector2f leftRayDir; // leftRayDir=direction-plane 
    float angle; //angle radian
}camera;

//met à jour les valeurs en fonction de l'angle
void evalCameraAngle(camera* c){
    c->direction.x=cos(c->angle); //cos et sin assure que la norme vaut 1
    c->direction.y=sin(c->angle);
    c->plane.x=c->direction.y*CAMERA_FOV;
    c->plane.y=-c->direction.x*CAMERA_FOV;
    c->leftRayDir.x=c->direction.x-c->plane.x;
    c->leftRayDir.y=c->direction.y-c->plane.y;
}

//retourne un pointer vers le pixel correspondant
void* gpuSurfaceGetPixel(SDL_Surface* gs,Uint32 x,Uint32 y){
    return gs->pixels+(y*gs->pitch+x*gs->format->BytesPerPixel);
}

/*
Copie une colonne de pixel d'une surface source vers une autre colonne sur la surface destination. 
Cette fonction donne la possibilité d'étirer ou rétrécir la colonne sur la surface de destination.
La fonction est utilisés pour le tracé des murs.
*/
void gpuSurfaceCopyColumn(SDL_Surface* src,SDL_Surface* dst,int src_column,int dst_column,int dst_height){
    int ystart,yend; //surface de destination
    int shift; //si dépassement sur la destination, de combien?
    int bpp; //byte per pixel

    bpp=src->format->BytesPerPixel;
    if (bpp!=dst->format->BytesPerPixel) return; //si les deux surfaces ont un format incompatible

    if (dst_height<0) dst_height=0; //imossible d'avoir une hauteur négative

    ystart=(dst->h>>1)-(dst_height>>1); 
    yend=ystart+dst_height;
    if (ystart<=0){ //si dépassement sur la surface de destination
        shift=-ystart;
        ystart=0;
        yend=dst->h;
    }else{
        shift=0; //sans dépassement
    }

    double step=((double)1/dst_height)*src->h; //décalage sur la source entre chaque pixel sur la destination
    double ysrc=shift*step; //surface source

    for (;ystart<yend;ystart++,ysrc+=step){
        memcpy(gpuSurfaceGetPixel(dst,dst_column,ystart),gpuSurfaceGetPixel(src,src_column,ysrc),bpp); //copie des pixels
    }
}

/*Sur la version CPU ce n'est pas un kernel cuda, c'est une simple fonction.
Cette fonction est responsable su tracé des murs sur l'écran.
explications sur ce site https://lodev.org/cgtutor/raycasting.html
*/
void wallCasting(camera* c,SDL_Surface* screen,SDL_Surface* wall,int i){
    float posX=c->position.x;
    float posY=c->position.y; //position

    float cameraX=((float)(i<<1))/screen->w;
    float rayDirX=c->leftRayDir.x+c->plane.x*cameraX; //direction du rayon étudié
    float rayDirY=c->leftRayDir.y+c->plane.y*cameraX;

    float rayDirX2=rayDirX*rayDirX;
    float rayDirY2=rayDirY*rayDirY;

    float rayDist=sqrt(rayDirX2+rayDirY2);  //norme du rayon

    float deltaDistX = rayDist/fabsf(rayDirX); 
    float deltaDistY = rayDist/fabsf(rayDirY);

    int stepX,stepY;
    float sideDistX,sideDistY;
    int side; //was a NS or a EW wall hit?

    int mapX=posX;
    int mapY=posY;

    float perpWallDist;
    float wallX; //where exactly the wall was hit

    if (rayDirX < 0){ //gérer la position au sein de la case
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
    //on avance jusqu'à taper contre un mur (la map étant entouré d'un mur, on est certain de s'arrêter)
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

    if(!side){
        perpWallDist = sideDistX-deltaDistX;
        wallX = posY + perpWallDist * rayDirY;
    }else{
        perpWallDist = sideDistY-deltaDistY;
        wallX = posX + perpWallDist * rayDirX;
    }
    
    int lineHeight = (float)screen->h/ perpWallDist*rayDist; //correction de l'effet fish eye.
    wallX -= floor(wallX);

    gpuSurfaceCopyColumn(wall,screen,wallX*wall->w,i,lineHeight); //copie de la ligne de pixel correspondante
}

/*Sur la version CPU ce n'est pas un kernel cuda, c'est une simple fonction.
Cette fonction est responsable su tracé du sol et du plafond l'écran.
explications sur ce site https://lodev.org/cgtutor/raycasting2.html.
Le tracé du sol ne dépend pas de la map. Il est dessiné sur tout l'écran puis recouvert du mur.
Contrairement au mur qui sont tracés colonne par colonne, le sol est tracé ligne par ligne
*/
void floorCasting(camera* c,SDL_Surface* screen,SDL_Surface* floor,SDL_Surface* ceil,int i){
    int bpp=screen->format->BytesPerPixel;
    
    int halfScreenHeight = screen->h>>1;
    float rowDistance = (float)halfScreenHeight / i;

    float floorStepX = rowDistance * (c->plane.x*2) / screen->w;
    float floorStepY = rowDistance * (c->plane.y*2) / screen->w;

    float floorX = c->position.x + rowDistance * c->leftRayDir.x; //position X sur le sol
    float floorY = c->position.y + rowDistance * c->leftRayDir.y; //position Y sur le sol

    //parcours horizontal
    for(int x = 0; x < screen->w; ++x){
        int tx = (int)(floor->w * (floorX - (int)floorX)) & (floor->w - 1); //position X dans le texture
        int ty = (int)(floor->h * (floorY - (int)floorY)) & (floor->h - 1); //position Y dans la texture

        floorX += floorStepX; //on décale les coordonnés en parcourant la ligne
        floorY += floorStepY;

        memcpy(gpuSurfaceGetPixel(screen,x,i+halfScreenHeight),gpuSurfaceGetPixel(floor,tx,ty),bpp);
        // le plafond est la symétrie axiale du sol par rapport au centre de l'écran
        memcpy(gpuSurfaceGetPixel(screen,x,halfScreenHeight-i-1),gpuSurfaceGetPixel(ceil,tx,ty),bpp); 
    }
}

//cette fonction charge un bitmap et la convertie vers le format de surface souhaité
SDL_Surface* loadBitMapFormat(char* restrict filename,SDL_PixelFormat* fmt){
    SDL_Surface* temp=SDL_LoadBMP(filename);
    SDL_Surface* s=SDL_ConvertSurface(temp,fmt,0);
    SDL_FreeSurface(temp);
    return s;
}

int main(int argc,char* argv[]){
    int relX;
    Uint8* keystate;
    SDL_Event event;
    struct timespec start, end; //chrono

    double nextX,nextY,stepX,stepY;

    if (argc==3){ //si la dimension de la fenêtre est passée en paramètre
        SCREEN_WIDTH=atoi(argv[1]);
        SCREEN_HEIGHT=atoi(argv[2]);
    }

    //initialisation de la SDL 1.2
    SDL_Init(SDL_INIT_VIDEO);
    SDL_Surface* screen = SDL_SetVideoMode(SCREEN_WIDTH, SCREEN_HEIGHT, 0, SDL_NOFRAME);
    SDL_ShowCursor(SDL_DISABLE);
    SDL_WM_GrabInput(SDL_GRAB_ON);

    //chargement des textures
    SDL_Surface* wall=loadBitMapFormat("brick.bmp",screen->format);
    SDL_Surface* floor=loadBitMapFormat("stone.bmp",screen->format);
    SDL_Surface* ceil=loadBitMapFormat("wood.bmp",screen->format);

    SDL_WM_SetIcon(floor,NULL); //set icon x)

    //position initiale de la caméra
    camera cam={.position=(vector2f){4,4},.angle=0};
    evalCameraAngle(&cam);

    while (1){
        clock_gettime(CLOCK_REALTIME, &start); // Chronomètre avant

        for(int y = 0; y < screen->h/2; y++) floorCasting(&cam,screen,floor,ceil,y);   //tracé su sol et du plafond 
        for (int i=0;i<SCREEN_WIDTH;i++) wallCasting(&cam,screen,wall,i); //tracé des murs

        clock_gettime(CLOCK_REALTIME, &end); // Chronomètre après
        double elapsed_time = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
        //printf("%f\n", elapsed_time * 1000); //affichage du temps écoulé
        //on lit les évènements
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
    //on libère les ressources et on quitte le programme
    SDL_FreeSurface(wall); 
    SDL_FreeSurface(floor);
    SDL_FreeSurface(ceil);
    
    SDL_FreeSurface(screen);
    SDL_Quit();

    return EXIT_SUCCESS;
}