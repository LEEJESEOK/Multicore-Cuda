#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "DS_timer.h"

#define MATRIX_I (1024)
#define MATRIX_J (1024 * 128)

/*
__global__ void matMul(int *a, int *b, int *c)
{
	const int NUM_THREAD_IN_BLOCK = blockDim.x * blockDim.y * blockDim.z;

 	int bID = blockIdx.z * (gridDim.y * gridDim.x * NUM_THREAD_IN_BLOCK) + blockIdx.y * (gridDim.x * NUM_THREAD_IN_BLOCK) + (blockIdx.x * (blockDim.x * blockDim.y * blockDim.z));
	int tID = bID + ((blockDim.y * blockDim.x) * threadIdx.z) + (blockDim.x * threadIdx.y) + threadIdx.x;
	
	c[tID] = a[tID] + b[tID];
}
*/

int main()
{
	// definition
	double **a, *b, *c;
	double **d_a, *d_b, *d_c;

	DS_timer timer(4);


	// init
	/*
	a : [MATRIX_I][MATRIX_J]
	b : [MATRIX_J]
	c : [MATRIX_I]
	*/
	a = (double **)malloc(sizeof(double *) * MATRIX_I);
	for(int i = 0; i < MATRIX_I; i++)
		a[i] = (double *)malloc(sizeof(double) * MATRIX_J);
	b = (double *)malloc(sizeof(double) * MATRIX_J); 
	c = (double *)malloc(sizeof(double) * MATRIX_I);

	for(int i = 0; i < MATRIX_I; i++)
		for(int j = 0; j < MATRIX_J; j++)
		{
			a[i][j] = rand() % 100;
		}
	for(int i = 0; i < MATRIX_J; i++)
		b[i] = rand() % 100;
	
	timer.initTimers();

	// end of init


	// serial version
	timer.onTimer(0);
	for(int i = 0; i < MATRIX_I; i++)
	{
		c[i] = 0;
		for(int j = 0; j < MATRIX_J; j++)
		{
			c[i] += a[i][j] * b[j];
		}	
	}
	timer.offTimer(0);

	// cuda version
        cudaMalloc((void **) &d_a, sizeof(double) * (MATRIX_I) * (MATRIX_J));
	cudaMalloc((void **) &d_b, sizeof(double) * MATRIX_J);
	cudaMalloc((void **) &d_c, sizeof(double) * MATRIX_I);

/*
	cudaMalloc((void ***) &d_a, sizeof(double *) * MATRIX_I);
	for(int i = 0; i < MATRIX_I; i++)
		cudaMalloc((void **) &(d_a[i]), sizeof(double) * MATRIX_J);
	cudaMalloc((void **) &d_b, sizeof(double) * MATRIX_J);
	cudaMalloc((void **) &d_c, sizeof(double) * MATRIX_I);
*/
/*
	timer.onTimer(1);
	for(int i = 0; i < MATRIX_I; i++)
		cudaMemcpy(d_a[i], a[i], sizeof(double) * MATRIX_J, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, b, sizeof(double) * MATRIX_J, cudaMemcpyHostToDevice);
	timer.offTimer(1);



	timer.onTimer(2);
	timer.offTimer(2);


	timer.onTimer(3);
	cudaMemcpy(c, d_c, sizeof(double) * MATRIX_I, cudaMemcpyDeviceToHost);
	timer.offTimer(3);
*/
	// check sequence
	
	
	timer.printTimer();


	cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);

	for(int i = 0; i < MATRIX_J; i++)
		delete[] a[i];
	delete[] a;
	delete[] b;
	delete[] c;

	return 0;
}
