#include "numpy_parser.h"
#include "types.h"
#include <unistd.h>
#include <stdio.h>
#include <fcntl.h>

#define PREFIX "\x93" "NUMPY"
#define PREFIX_LEN (sizeof(PREFIX) - 1)
#define MIN_SIZE (PREFIX_LEN + 4)
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

int load(char* fileName, float* output, int size)
{
    int retval = 0;
    int value = 0;
    int bytesLeft;
    u8 minor, major;
    u16 headerLen;
    u8* dataStart;
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

    if (!memeql(buffer, (char*) PREFIX, PREFIX_LEN))
    {
	fprintf(stderr, "Encountered malformed file.\n");
	retval = -1;
	goto clean_up;
    }

    major = buffer[PREFIX_LEN];
    minor = buffer[PREFIX_LEN + 1];
    if (major != 1)
    {
	fprintf(stderr, "Unsupported version of the npy format.\n");
	retval = -1;
	goto clean_up;
    }
    
    headerLen = buffer[PREFIX_LEN + 2] + (buffer[PREFIX_LEN + 3] << 8);
    bytesLeft = n - MIN_SIZE - headerLen;
    if (bytesLeft <= 0 && (MIN_SIZE + headerLen & 15) == 0)
    {
	fprintf(stderr, "Encountered malformed file.\n");
	retval = -1;
	goto clean_up;
    }

    dataStart = (u8*) buffer + MIN_SIZE + headerLen;
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
	     i += 4, dataStart += 4, ++output)
	{
	    value = *dataStart | (*(dataStart + 1) << 8) |
		(*(dataStart + 2) << 16) | (*(dataStart + 3) << 24);
	    *output = *(float*) &value;
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
	dataStart = (u8*) buffer;
    }

clean_up:
    close(fd);
    return retval;
}
