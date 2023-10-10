BUILD_FOLDER=build
ASSETS_FOLDER=assets
SRC_FOLDER=src
CUDA_FOLDER=/opt/cuda
TMP_FOLDER=$(BUILD_FOLDER)/hip

OUT=main
SRC_C=main.c
SRC_CUDA=main2.cu
SRC_HEADER=main.h

CFLGAGS=-Wall
LFLAGS=-lSDL

SCREEN_WIDTH=1920
SCREEN_HEIGHT=1080

all: cpu

amd: hip

nvidia: cuda

clean:
	@echo "cleaning build..."
	@rm -rf $(BUILD_FOLDER)
	@if [ -d $(TMP_FOLDER) ]; then \
		rm -rf $(TMP_FOLDER); \
	fi

init_build_folder:
	@echo "init build folder..."
	@if [ ! -d $(BUILD_FOLDER) ]; then \
		mkdir $(BUILD_FOLDER); \
		cp $(ASSETS_FOLDER)/* $(BUILD_FOLDER); \
	fi

cpu: init_build_folder
	@echo "compiling source file..."
	@gcc $(SRC_FOLDER)/$(SRC_C) $(CFLAGS) $(LFLAGS) -lm -O5 -o $(BUILD_FOLDER)/$(OUT)

cuda: init_build_folder
	@echo "compiling cuda file..."
	@nvcc $(SRC_FOLDER)/$(SRC_CUDA) $(LFLAGS) -o $(BUILD_FOLDER)/$(OUT)

hip: init_build_folder
	@echo "hipify cuda file..."
	@mkdir -p $(TMP_FOLDER)
	@cp $(SRC_FOLDER)/$(SRC_HEADER) $(TMP_FOLDER)
	@hipify-clang $(SRC_FOLDER)/$(SRC_CUDA) --cuda-path=$(CUDA_FOLDER) -o $(TMP_FOLDER)/$(OUT).hip
	@echo "compiling hip file..."
	@hipcc $(TMP_FOLDER)/$(OUT).hip $(LFLAGS) -o $(BUILD_FOLDER)/$(OUT)

run: 
	@echo "running executable..."
	@cd $(BUILD_FOLDER) && ./$(OUT) $(SCREEN_WIDTH) $(SCREEN_HEIGHT) > result.txt