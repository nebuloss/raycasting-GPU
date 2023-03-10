all: compile run

compile:
	gcc main.c -g -lSDL -lm -o main

run:
	./main