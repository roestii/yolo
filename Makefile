yolo: yolo.o
	gcc -o yolo yolo.o

yolo.o: src/yolo.cpp src/types.h src/test_data.h
	gcc -c src/yolo.cpp

