CFLAGS ?= -g
CC = gcc

test: test.o layers.o f2x2_3x3_convolution.o f3x3_2x2_convolution.o algebra.o arena.o numpy_parser.o
	$(CC) $(CFLAGS) -o test test.o algebra.o f2x2_3x3_convolution.o f3x3_2x2_convolution.o layers.o arena.o numpy_parser.o

yolo: yolo.o algebra.o f2x2_3x3_convolution.o f3x3_2x2_convolution.o layers.o
	$(CC) $(CFLAGS) -o yolo yolo.o algebra.o f2x2_3x3_convolution.o f3x3_2x2_convolution.o layers.o

test.o: src/test.cpp src/types.h src/arena.h src/layers.h
	$(CC) $(CFLAGS) -c src/test.cpp

numpy_parser.o: src/numpy_parser.cpp src/numpy_parser.h src/types.h
	$(CC) $(CFLAGS) -c src/numpy_parser.cpp

arena.o: src/arena.cpp src/arena.h src/types.h
	$(CC) $(CFLAGS) -c src/arena.cpp

yolo.o: src/yolo.cpp src/types.h
	$(CC) $(CFLAGS) -c src/yolo.cpp

algebra.o: src/algebra.cpp src/algebra.h src/types.h
	$(CC) $(CFLAGS) -c src/algebra.cpp

layers.o: src/layers.cpp src/layers.h src/types.h src/f2x2_3x3_convolution.h src/f3x3_2x2_convolution.h
	$(CC) $(CFLAGS) -c src/layers.cpp

f2x2_3x3_convolution.o: src/f2x2_3x3_convolution.cpp src/f2x2_3x3_convolution.h src/types.h src/algebra.h
	$(CC) $(CFLAGS) -c src/f2x2_3x3_convolution.cpp

f3x3_2x2_convolution.o: src/f3x3_2x2_convolution.cpp src/f3x3_2x2_convolution.h src/types.h src/algebra.h
	$(CC) $(CFLAGS) -c src/f3x3_2x2_convolution.cpp

clean:
	rm *.o yolo
