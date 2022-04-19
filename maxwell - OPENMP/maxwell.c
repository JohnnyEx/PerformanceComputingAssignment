#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <unistd.h>

#include "args.h"
#include "vtk.h"
#include "data.h"
#include "setup.h"

#include <time.h>
#include <omp.h>

#include <papi.h>

/**
 * @brief Update the magnetic and electric fields. The magnetic fields are updated for a half-time-step. The electric fields are updated for a full time-step.
 * 
 */
void update_fields() {

	// constants;
	double dtx = (dt / dx);
	double dty = (dt / dy);
	double dtdyepsmu = dt / (dy * eps * mu);
	double dtdxepsmu = dt / (dx * eps * mu);

	int i = 0;
	int j = 0;
	#pragma omp parallel for private (i,j) schedule(static)
	for (i = 0; i < Bz_size_x; i++) {
		for (j = 0; j < Bz_size_y; j++) {
			Bz[i][j] = Bz[i][j] - dtx * (Ey[i+1][j] - Ey[i][j])
				                + dty * (Ex[i][j+1] - Ex[i][j]);
		}
	}

	#pragma omp parallel for private (i,j) schedule(static)
	for (i = 0; i < Ex_size_x; i++) {
		for (j = 1; j < Ex_size_y-1; j++) {
			Ex[i][j] = Ex[i][j] + dtdyepsmu * (Bz[i][j] - Bz[i][j-1]);
		}
	}

	#pragma omp parallel for private(i,j) schedule(static)
	for (i = 1; i < Ey_size_x-1; i++) {
		for (j = 0; j < Ey_size_y; j++) {
			Ey[i][j] = Ey[i][j] - dtdxepsmu * (Bz[i][j] - Bz[i-1][j]);
		}
	}
}

/**
 * @brief Apply boundary conditions
 * 
 */
void apply_boundary() {
	int i = 0;
	int j = 0;

	#pragma omp parallel for schedule(static)
	for (i = 0; i < Ex_size_x; i++) {
		Ex[i][0] = -Ex[i][1];
		Ex[i][Ex_size_y-1] = -Ex[i][Ex_size_y-2];
	}

	#pragma omp parallel for schedule(static)
	for (j = 0; j < Ey_size_y; j++) {
		Ey[0][j] = -Ey[1][j];
		Ey[Ey_size_x-1][j] = -Ey[Ey_size_x-2][j];
	}
}

/**
 * @brief Resolve the Ex, Ey and Bz fields to grid points and sum the magnitudes for output
 * 
 * @param E_mag The returned total magnitude of the Electric field (E)
 * @param B_mag The returned total magnitude of the Magnetic field (B) 
 */
void resolve_to_grid(double *E_mag, double *B_mag) {
	double e_mag = 0.0;
	double b_mag = 0.0;

	int i, j;
	#pragma omp parallel for private (i,j) reduction (+:e_mag) schedule(static)
	for (i = 1; i < E_size_x-1; i++) {
		for (j = 1; j < E_size_y-1; j++) {
			E[i][j][0] = (Ex[i-1][j] + Ex[i][j]) / 2.0;
			E[i][j][1] = (Ey[i][j-1] + Ey[i][j]) / 2.0;
			//E[i][j][2] = 0.0; // in 2D we don't care about this dimension
			e_mag += sqrt((E[i][j][0] * E[i][j][0]) + (E[i][j][1] * E[i][j][1]));
		}
	}
	
	#pragma omp parallel for private (i,j) reduction (+:b_mag) schedule(static)
	for (i = 1; i < B_size_x-1; i++) {
		for (j = 1; j < B_size_y-1; j++) {
			//B[i][j][0] = 0.0; // in 2D we don't care about these dimensions
			//B[i][j][1] = 0.0;
			B[i][j][2] = (Bz[i-1][j] + Bz[i][j] + Bz[i][j-1] + Bz[i-1][j-1]) / 4.0;

			b_mag += sqrt(B[i][j][2] * B[i][j][2]);
		}
	}

	*E_mag = e_mag;
	*B_mag = b_mag;
}

/**
 * @brief The main routine that sets up the problem and executes the timestepping routines
 * 
 * @param argc The number of arguments passed to the program
 * @param argv An array of the arguments passed to the program
 * @return int The return value of the application
 */
int main(int argc, char *argv[]) {

	// omp threads and stuff
	int const nMaxOMPThreads = omp_get_max_threads();
	int const nNumProcessors = omp_get_num_procs();

	printf("     OMP system reports following parameters: \n");
	printf("         max number of threads    = %6d \n", nMaxOMPThreads);
	printf("         number of computer nodes = %6d   ", nNumProcessors);
	printf("\n\n");	
	// end omp
	
	// Papi code ffs
	int retval;
	retval=PAPI_library_init(PAPI_VER_CURRENT);
	if (retval!=PAPI_VER_CURRENT) {
		printf("Error initializing PAPI! %s\n",	PAPI_strerror(retval));
		return 0;
	}

	int eventset=PAPI_NULL;

	retval=PAPI_create_eventset(&eventset);
	if (retval!=PAPI_OK) {
		printf("Error creating eventset! %s\n", PAPI_strerror(retval));
	}

	retval=PAPI_add_named_event(eventset,"PAPI_TOT_CYC");
	if (retval!=PAPI_OK) {
		printf("Error adding PAPI_TOT_CYC: %s\n",PAPI_strerror(retval));
	}
	//end papi initialization

	// time starting
	clock_t begin = clock();

	// papi start
	long long count;

	PAPI_reset(eventset);
	retval=PAPI_start(eventset);
	if (retval!=PAPI_OK) {
		printf("Error starting CUDA: %s\n",PAPI_strerror(retval));
	}

	set_defaults();
	parse_args(argc, argv);
	setup();

	printf("Running problem size %f x %f on a %d x %d grid.\n", lengthX, lengthY, X, Y);
	
	if (verbose) print_opts();
	
	allocate_arrays();

	problem_set_up();

	// start at time 0
	double t = 0.0;
	int i = 0;
	while (i < steps) {
		apply_boundary();
		update_fields();

		t += dt;

		if (i % output_freq == 0) {
			double E_mag, B_mag;
			resolve_to_grid(&E_mag, &B_mag);
			printf("Step %8d, Time: %14.8e (dt: %14.8e), E magnitude: %14.8e, B magnitude: %14.8e\n", i, t, dt, E_mag, B_mag);

			if ((!no_output) && (enable_checkpoints))
				write_checkpoint(i);
		}

		i++;
	}

	double E_mag, B_mag;
	resolve_to_grid(&E_mag, &B_mag);

	printf("Step %8d, Time: %14.8e (dt: %14.8e), E magnitude: %14.8e, B magnitude: %14.8e\n", i, t, dt, E_mag, B_mag);
	printf("Simulation complete.\n");

	// time stop
	clock_t end = clock();
    // calc the time;
	double time_spent = (double)(end-begin) / CLOCKS_PER_SEC / omp_get_num_threads();
	printf("Time spent for this execution: %lf", time_spent);

	// end papii
	retval=PAPI_stop(eventset,&count);
	if (retval!=PAPI_OK) {
		printf("Error stopping:  %s\n", PAPI_strerror(retval));
	}
	else {
		printf("Measured %lld cycles\n",count);
	}

	PAPI_cleanup_eventset(eventset);
    PAPI_destroy_eventset(&eventset);
	// end

	if (!no_output) 
		write_result();

	free_arrays();

	exit(0);
}


