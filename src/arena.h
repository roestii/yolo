#ifndef ARENA_H
#define ARENA_H

#include "types.h"

struct arena
{
    uintptr start;
    size_t offset;
    size_t len;
};

void initArena(arena*, uintptr, size_t);
void* pushSize(arena*, size_t);

#endif
