BUILD_FOLDER=build
ASSETS_FOLDER=assets
SRC_FOLDER=src
CUDA_FOLDER=/opt/cuda

OUT=main
SRC_C=main.c
SRC_CUDA=main.cu

SCREEN_WIDTH=1920
SCREEN_HEIGHT=1440

all: cpu run

amd: hip run

nvidia: cuda run

clean:
	@echo "cleaning build..."
	@rm -rf $(BUILD_FOLDER)

init_build_folder:
	@echo "init build folder..."
	@if [ ! -d $(BUILD_FOLDER) ]; then \
		mkdir $(BUILD_FOLDER); \
		cp $(ASSETS_FOLDER)/* $(BUILD_FOLDER); \
	fi

cpu: init_build_folder
	@echo "compiling source file..."
	@gcc $(SRC_FOLDER)/$(SRC_C) -lSDL -lm -O5 -o $(BUILD_FOLDER)/$(OUT)

cuda: init_build_folder
	@echo "compiling cuda file..."
	@nvcc $(SRC_FOLDER)/$(SRC_CUDA) -lSDL -o $(BUILD_FOLDER)/$(OUT)

hip: init_build_folder
	@echo "hipify cuda file..."
	@hipify-clang $(SRC_FOLDER)/$(SRC_CUDA) --cuda-path=$(CUDA_FOLDER) -o $(BUILD_FOLDER)/$(OUT).hip
	@echo "compiling hip file..."
	@hipcc $(BUILD_FOLDER)/$(OUT).hip -lSDL -o $(BUILD_FOLDER)/$(OUT)

run: 
	@echo "running executable..."
	@cd $(BUILD_FOLDER) && ./$(OUT) $(SCREEN_WIDTH) $(SCREEN_HEIGHT)