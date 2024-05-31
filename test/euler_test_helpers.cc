#include <stdlib.h>

double rand_01() {
    int range = 1000000000;
    return (rand() % range) / 1e9;
}
