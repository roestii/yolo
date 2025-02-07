#include "arena.h"

void initArena(arena* a, uintptr start, size_t len)
{
    a->offset = 0;
    a->start = start;
    a->len = len;
}

void* pushSize(arena* a, size_t size)
{
    assert(a->offset + size <= a->len)
    uintptr res = a->start + a->offset;
    a->offset += size;
    return (void*) res;
}
