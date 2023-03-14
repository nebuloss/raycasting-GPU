all: compile_cpu run

gpu: compile_gpu run

compile_cpu:
	gcc main.c -lSDL -lm -O5 -o main

compile_gpu:
	nvcc main.cu -lSDL -o main

run:
	./main 1440 1080