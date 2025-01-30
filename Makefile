CFLAGS = -g
CC = clang

yolo: yolo.o
	$(CC) $(CFLAGS) -o yolo yolo.o

yolo.o: src/yolo.cpp src/types.h src/test_data.h
	$(CC) $(CFLAGS) -c src/yolo.cpp
clean:
	rm *.o yolo
