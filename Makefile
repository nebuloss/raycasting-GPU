all: compile run

compile:
	gcc main.c -lSDL -o main

run:
	./main