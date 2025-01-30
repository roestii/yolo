CFLAGS = -g
CC = clang

yolo: yolo.o algebra.o image.o
	$(CC) $(CFLAGS) -o yolo yolo.o image.o algebra.o

yolo.o: src/yolo.cpp src/types.h src/test_data.h src/algebra.h src/image.h
	$(CC) $(CFLAGS) -c src/yolo.cpp

algebra.o: src/algebra.cpp src/algebra.h src/types.h
	$(CC) $(CFLAGS) -c src/algebra.cpp

image.o: src/image.cpp src/image.h src/types.h
	$(CC) $(CFLAGS) -c src/image.cpp

clean:
	rm *.o yolo
