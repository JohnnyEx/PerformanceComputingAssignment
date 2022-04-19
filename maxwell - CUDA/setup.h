#ifndef SETUP_H
#define SETUP_H

void set_defaults();
void setup();
void allocate_arrays();
void free_arrays();
__global__ void problem_set_up(variables m_variables, arrays m_arrays);

#endif