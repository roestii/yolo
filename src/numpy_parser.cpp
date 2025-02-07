#include "numpy_parser.h"
#include "types.h"
#include <unistd.h>
#include <stdio.h>

#define PREFIX "\x93" "NUMPY"
#define MIN_SIZE sizeof(PREFIX) + 4
#define Align16(x) (x + 15) & ~15

static int memeql(char* a, char* b, int size)
{
    for (int i = 0;
	 i < size;
	 ++i, ++a, ++b)
    {
	if (*a != *b)
	{
	    return 0;
	}
    }
    return 1;
}

int load(char* fileName, float* buffer, int size)
{
    int retval = 0;
    u32 value;
    int bytesLeft;
    u8 minor, major;
    u16 headerLen;
    char* dataStart;
    int fd = open(fileName, O_RDONLY);
    if (fd == -1)
    {
	fprintf(stderr, "Cannot open file.\n");
	return -1;
    }
    
    char buffer[1024];
    ssize_t n = read(fd, buffer, sizeof(buffer));
    if (n < MIN_SIZE)
    {
	fprintf(stderr, "Encountered malformed file.\n");
	retval = -1;
	goto clean_up;
    }

    if (!memeql(buffer, PREFIX, sizeof(PREFIX)))
    {
	fprintf(stderr, "Encountered malformed file.\n");
	retval = -1;
	goto clean_up;
    }

    major = buffer[sizeof(PREFIX)];
    minor = buffer[sizeof(PREFIX) + 1];
    if (major != 1)
    {
	fprintf(stderr, "Unsupported version of the npy format.\n");
	retval = -1;
	goto clean_up;
    }
    
    headerLen = buffer[sizeof(PREFIX) + 2] + buffer[sizeof(PREFIX) + 3] << 8;
    bytesLeft = n - MIN_SIZE - headerLen;
    if (bytesLeft <= 0 && MIN_SIZE + headerLen & 15 == 0)
    {
	fprintf(stderr, "Encountered malformed file.\n");
	retval = -1;
	goto clean_up;
    }

    dataStart = buffer + MIN_SIZE + headerLen;
    if (*(dataStart - 1) != '\n')
    {
	fprintf(stderr, "Encountered malformed file.\n");
	retval = -1;
	goto clean_up;
    }

    for (;;)
    {
	for (int i = 0;
	     i < bytesLeft;
	     i += 4, dataStart += 4, ++buffer)
	{
	    value = *dataStart + *(dataStart + 1) << 8 +
		*(dataStart + 2) << 16 + *(dataStart + 3) << 24;
	    *buffer = *(float*) value;
	}

	bytesLeft = read(fd, buffer, sizeof(buffer));
	if (bytesLeft == 0)
	{
	    break;
	}
	else if (bytesLeft < 0)
	{
	    fprintf(stderr, "Error while reading file.\n");
	}
	dataStart = buffer;
    }

clean_up:
    close(fd);
    return retval;
}
