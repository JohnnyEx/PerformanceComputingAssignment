#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <unistd.h>

#include "args.h"
#include "vtk.h"
#include "data.h"
#include "setup.h"

#include <mpi.h>

/**
 * @brief Update the magnetic and electric fields. The magnetic fields are updated for a half-time-step. The electric fields are updated for a full time-step.
 * 
 */
void update_fields(MPI_Datatype Ex_col, MPI_Datatype Ey_colm, MPI_Datatype Ez_col, int size, int rank, int left, int right) {
	
	MPI_Barrier(MPI_COMM_WORLD);
	MPI_Sendrecv(Ey[0], 1, Ey_colm, left, 13, Ey[Ey_size_x-1], 1, Ey_colm, right, 13, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

	for (int i = 1; i < Bz_size_x + 1; i++) {
		for (int j = 0; j < Bz_size_y; j++) {
			Bz[i][j] = Bz[i][j] - (dt / dx) * (Ey[i+1][j] - Ey[i][j])
				                + (dt / dy) * (Ex[i][j+1] - Ex[i][j]);
		}
	}

	for (int i = 1; i < Ex_size_x + 1; i++) {
		for (int j = 1; j < Ex_size_y-1; j++) {
			Ex[i][j] = Ex[i][j] + (dt / (dy * eps * mu)) * (Bz[i][j] - Bz[i][j-1]);
		}
	}

	MPI_Barrier(MPI_COMM_WORLD);
	MPI_Sendrecv(Bz[Bz_size_x], 1, Ez_col, right, 13, Bz[0], 1, Ez_col, left, 13, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    for (int i = 0; i < Ey_size_x-1; i++) {
        for (int j = 0; j < Ey_size_y; j++) {
            Ey[i][j] = Ey[i][j] - (dt / (dx * eps * mu)) * (Bz[i+1][j] - Bz[i][j]);
		}
	}

	MPI_Barrier(MPI_COMM_WORLD);
	MPI_Sendrecv(Ex[Ex_size_x], 1, Ex_col, right, 13, Ex[0], 1, Ex_col, left, 13, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
}

/**
 * @brief Apply boundary conditions
 * 
 */
void apply_boundary(int rank, int size) {
	for (int i = 1; i < Ex_size_x + 1; i++) {
		Ex[i][0] = -Ex[i][1];
		Ex[i][Ex_size_y-1] = -Ex[i][Ex_size_y-2];
	}

	for (int j = 0; j < Ey_size_y; j++) {
		if (rank == 0) Ey[0][j] = -Ey[1][j];
		if (rank == size - 1) Ey[Ey_size_x-1][j] = -Ey[Ey_size_x-2][j];
	}
}

/**
 * @brief Resolve the Ex, Ey and Bz fields to grid points and sum the magnitudes for output
 * 
 * @param E_mag The returned total magnitude of the Electric field (E)
 * @param B_mag The returned total magnitude of the Magnetic field (B) 
 */
void resolve_to_grid(double *E_mag, double *B_mag, int rank, int size) {
	*E_mag = 0.0;
	*B_mag = 0.0;

    for (int i = rank != 0 ? 0 : 1; i < E_size_x-1; i++) {
        for (int j = 1; j < E_size_y-1; j++) {
            E[i][j][0] = (Ex[i][j] + Ex[i+1][j]) / 2.0;
            E[i][j][1] = (Ey[i][j-1] + Ey[i][j]) / 2.0;
            *E_mag += sqrt((E[i][j][0] * E[i][j][0]) + (E[i][j][1] * E[i][j][1]));
        }
    }

    for (int i = rank == 0 ? 1 : 0; i < B_size_x-1; i++) {
        for (int j = 1; j < B_size_y - 1; j++) {
            B[i][j][2] = (Bz[i][j] + Bz[i + 1][j] + Bz[i + 1][j - 1] + Bz[i][j - 1]) / 4.0;
            *B_mag += sqrt(B[i][j][2] * B[i][j][2]);
        }
    }
	// perform a sum reduction to help calculate the global mean value
}

/**
 * @brief The main routine that sets up the problem and executes the timestepping routines
 * 
 * @param argc The number of arguments passed to the program
 * @param argv An array of the arguments passed to the program
 * @return int The return value of the application
 */
int main(int argc, char *argv[]) {
	// Initialize MPI
	int rank, size;
	MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	set_defaults();
	parse_args(argc, argv);
	setup();

	if(rank == 0) printf("Running problem size %f x %f on a %d x %d grid.\n", lengthX, lengthY, X, Y);
	
	if (verbose) print_opts();
	
	allocate_arrays(rank, size);

	problem_set_up(rank, size);

	// spray and pray
	MPI_Datatype Ex_col, Ey_col, Bz_col;
	MPI_Type_vector(1, Ex_size_y, Ex_size_y, MPI_DOUBLE, &Ex_col);
	MPI_Type_commit(&Ex_col);
	MPI_Type_vector(1, Ey_size_y, Ey_size_y, MPI_DOUBLE, &Ey_col);
	MPI_Type_commit(&Ey_col);
	MPI_Type_vector(1, Bz_size_y, Bz_size_y, MPI_DOUBLE, &Bz_col);
	MPI_Type_commit(&Bz_col);
	int left = rank-1 < 0 ? MPI_PROC_NULL : rank-1;
	int right = rank+1 >= size ? MPI_PROC_NULL: rank+1;

	// start at time 0
	double t = 0.0;
	int i = 0;
	while (i < steps) {
		apply_boundary(rank, size);
		update_fields(Ex_col, Ey_col, Bz_col, rank, size, left, right);

		t += dt;

		if (i % output_freq == 0) {
			double E_mag, B_mag;
			resolve_to_grid(&E_mag, &B_mag, rank, size);
			// waiting for everyone here and reduce them
			MPI_Barrier(MPI_COMM_WORLD);
			MPI_Allreduce(MPI_IN_PLACE, &B_mag, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
			MPI_Allreduce(MPI_IN_PLACE, &E_mag, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
			if (rank == 0) printf("Step %8d, Time: %14.8e (dt: %14.8e), E magnitude: %14.8e, B magnitude: %14.8e\n", i, t, dt, E_mag, B_mag);

			if ((!no_output) && (enable_checkpoints) && rank == 0)
				write_checkpoint(i);
		}

		i++;
	}

	double E_mag, B_mag;
	resolve_to_grid(&E_mag, &B_mag, rank, size);
	// waiting for everyone here
	MPI_Barrier(MPI_COMM_WORLD);
	MPI_Allreduce(MPI_IN_PLACE, &B_mag, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
	MPI_Allreduce(MPI_IN_PLACE, &E_mag, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

	if (rank == 0) printf("Step %8d, Time: %14.8e (dt: %14.8e), E magnitude: %14.8e, B magnitude: %14.8e\n", i, t, dt, E_mag, B_mag);
	if (rank == 0) printf("Simulation complete.\n");

	if (!no_output && rank == 0) 
		write_result();

	free_arrays();
	MPI_Type_free(&Ex_col);
	MPI_Type_free(&Ey_col);
	MPI_Type_free(&Bz_col);
	
	// Finalizing the MPI - which should return success / failure
	MPI_Finalize();
	
	exit(0);
}


