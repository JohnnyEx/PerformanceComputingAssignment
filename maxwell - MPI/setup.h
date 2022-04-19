#ifndef SETUP_H
#define SETUP_H

void set_defaults();
void setup();
void allocate_arrays(int rank, int size);
void free_arrays();
void problem_set_up(int rank, int size);

#endif