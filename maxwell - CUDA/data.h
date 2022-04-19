#ifndef DATA_H
#define DATA_H

struct constants {
    const double c;
    const double mu;
    const double eps;
    const double cfl;
};

struct variables {
    double lengthX;
    double lengthY;
    int X;
    int Y;
    double dx;
    double dy;
    double dt;
};

// Time to run for / or number of steps
extern double T;
extern int steps;

// Cuda variable
struct cudaGraph {
    int grid_x;
    int grid_y;
    int block_x;
    int block_y;
};

extern struct constants m_constants;
extern struct variables m_variables;
extern struct cudaGraph graph;

// x = Ex values
// o = Ey values
// * = Bz values
// + = E and B values
//
//    +---x---+---x---+---x---+---x---+---x---+
//    |       |       |       |       |       |
//    o   *   o   *   o   *   o   *   o   *   o
//    |       |       |       |       |       |
//    +---x---+---x---+---x---+---x---+---x---+
//    |       |       |       |       |       |
//    o   *   o   *   o   *   o   *   o   *   o
//    |       |       |       |       |       |
//    +---x---+---x---+---x---+---x---+---x---+
// ^  |       |       |       |       |       |
// |  o   *   o   *   o   *   o   *   o   *   o
// y  |       |       |       |       |       |
//    +---x---+---x---+---x---+---x---+---x---+
//(0,0)  x -> 

struct arrays {
    int Ex_size_x, Ex_size_y;
    double * Ex;
    size_t ex_pitch;
    int Ey_size_x, Ey_size_y;
    double * Ey;
    size_t ey_pitch;
    int Bz_size_x, Bz_size_y;
    double * Bz;
    size_t bz_pitch;
    int E_size_x, E_size_y, E_size_z;
    double * E;
    size_t e_pitch;
    int B_size_x, B_size_y, B_size_z;
    double * B;
    size_t b_pitch;
};

extern arrays m_arrays;

extern double *** host_E;
extern double *** host_B;

void alloc_2d_array(int m, int n, double **array, size_t *pitch);
void free_2d_array(double *array);
void alloc_3d_cuda_array(int m, int n, int o, double **array, size_t *pitch);
void free_3d_cuda_array(double *array);
double ***alloc_3d_array(int m, int n, int o);
void free_3d_array(double*** array);

#endif