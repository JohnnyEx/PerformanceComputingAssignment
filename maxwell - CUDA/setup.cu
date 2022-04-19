#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <unistd.h>

#include "vtk.h"
#include "data.h"
#include "setup.h"

/**
 * @brief Set up some default values before arguments have been loaded
 * 
 */
void set_defaults() {
	m_variables.lengthX = 1.0;
	m_variables.lengthY = 1.0;

	m_variables.X = 4000;
	m_variables.Y = 4000;

    graph.grid_x = 10;
    graph.grid_y = 10;
    graph.block_x = 50;
    graph.block_y = 50;

	T = 1.6e-9;

	set_default_base();
}

/**
 * @brief Set up some of the values required for computation after arguments have been loaded
 * 
 */
void setup() {
	m_variables.dx = m_variables.lengthX / m_variables.X;
	m_variables.dy = m_variables.lengthY / m_variables.Y;

	m_variables.dt = m_constants.cfl * (m_variables.dx > m_variables.dy ? m_variables.dx : m_variables.dy) / m_constants.c;
	
	if (steps == 0) // only set this if steps hasn't been specified
		steps = (int) (T / m_variables.dt);
}

/**
 * @brief Allocate all of the arrays used for computation
 * 
 */
void allocate_arrays() {
    m_arrays.Ex_size_x = m_variables.X; m_arrays.Ex_size_y = m_variables.Y+1;
    alloc_2d_array(m_variables.X, m_variables.Y+1, &m_arrays.Ex, &m_arrays.ex_pitch);
    m_arrays.Ey_size_x = m_variables.X+1; m_arrays.Ey_size_y = m_variables.Y;
    alloc_2d_array(m_variables.X+1, m_variables.Y, &m_arrays.Ey, &m_arrays.ey_pitch);
    m_arrays.Bz_size_x = m_variables.X; m_arrays.Bz_size_y = m_variables.Y;
    alloc_2d_array(m_variables.X, m_variables.Y, &m_arrays.Bz, &m_arrays.bz_pitch);

    m_arrays.E_size_x = m_variables.X+1; m_arrays.E_size_y = m_variables.Y+1; m_arrays.E_size_z = 3;
    alloc_3d_cuda_array(m_arrays.E_size_x, m_arrays.E_size_y, m_arrays.E_size_z, &m_arrays.E, &m_arrays.e_pitch);
    host_E = alloc_3d_array(m_arrays.E_size_x, m_arrays.E_size_y, m_arrays.E_size_z);
    m_arrays.B_size_x = m_variables.X+1; m_arrays.B_size_y = m_variables.Y+1; m_arrays.B_size_z = 3;
    alloc_3d_cuda_array(m_arrays.B_size_x, m_arrays.B_size_y, m_arrays.B_size_z, &m_arrays.B, &m_arrays.b_pitch);
    host_B = alloc_3d_array(m_arrays.B_size_x, m_arrays.B_size_y, m_arrays.B_size_z);
}

/**
 * @brief Free all of the arrays used for the computation
 * 
 */
void free_arrays() {
	free_2d_array(m_arrays.Ex);
	free_2d_array(m_arrays.Ey);
	free_2d_array(m_arrays.Bz);
    free_3d_cuda_array(m_arrays.E);
    free_3d_cuda_array(m_arrays.B);
	free_3d_array(host_E);
	free_3d_array(host_B);
}

/**
 * @brief Set up a guassian to curve around the centre
 * 
 */
__global__ void problem_set_up(variables m_variables, arrays m_arrays) {
    double xcen = m_variables.lengthX / 2.0;
    double ycen = m_variables.lengthY / 2.0;

    for (int i = 0; i < m_arrays.Ex_size_x; i++ ) {
        for (int j = 0; j < m_arrays.Ex_size_y; j++) {
            double xcoord = (i - xcen) * m_variables.dx;
            double ycoord = j * m_variables.dy;
            double rx = xcen - xcoord;
            double ry = ycen - ycoord;
            double rlen = sqrt(rx*rx + ry*ry);
            double tx = (rlen == 0) ? 0 : ry / rlen;
            double mag = exp(-400.0 * (rlen - (m_variables.lengthX / 4.0)) * (rlen - (m_variables.lengthY / 4.0)));
            m_arrays.Ex[i * m_arrays.ex_pitch + j] = mag * tx;
        }
    }
    for (int i = 0; i < m_arrays.Ey_size_x; i++ ) {
        for (int j = 0; j < m_arrays.Ey_size_y; j++) {
            double xcoord = i * m_variables.dx;
            double ycoord = (j - ycen) * m_variables.dy;
            double rx = xcen - xcoord;
            double ry = ycen - ycoord;
            double rlen = sqrt(rx*rx + ry*ry);
            double ty = (rlen == 0) ? 0 : -rx / rlen;
            double mag = exp(-400.0 * (rlen - (m_variables.lengthY / 4.0)) * (rlen - (m_variables.lengthY / 4.0)));
            m_arrays.Ey[i*m_arrays.ey_pitch + j] = mag * ty;
        }
    }
}
