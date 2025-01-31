CFLAGS = -g
CC = clang

yolo: yolo.o algebra.o
	$(CC) $(CFLAGS) -o yolo yolo.o algebra.o 

yolo.o: src/yolo.cpp src/types.h src/test.h src/layer_definition.h src/layers.h
	$(CC) $(CFLAGS) -c src/yolo.cpp

algebra.o: src/algebra.cpp src/algebra.h src/types.h
	$(CC) $(CFLAGS) -c src/algebra.cpp

clean:
	rm *.o yolo
