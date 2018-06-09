#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "DS_timer.h"

#define MATRIX_I (1024)
#define MATRIX_J (1024 * 64)
#define MATRIX_K (1)

__global__ void matMul(double *a, double *b, double *c)
{
	const int NUM_THREAD_IN_BLOCK = blockDim.x * blockDim.y * blockDim.z;

 	int bID = blockIdx.z * (gridDim.y * gridDim.x * NUM_THREAD_IN_BLOCK) + blockIdx.y * (gridDim.x * NUM_THREAD_IN_BLOCK) + (blockIdx.x * (blockDim.x * blockDim.y * blockDim.z));
	int tID = bID + ((blockDim.y * blockDim.x) * threadIdx.z) + (blockDim.x * threadIdx.y) + threadIdx.x;

	
	for(int i = 0; i < MATRIX_J; i++)
		c[tID] += a[(tID * MATRIX_J) + i] * b[i];	
}


int main()
{
	// definition
	double *a, *b, *c1, *c2;
	double *d_a, *d_b, *d_c;

	bool result;

	DS_timer timer(4);


	// init
	/*
	a : [MATRIX_I][MATRIX_J]
	b : [MATRIX_J] * 1
	c : [MATRIX_I] * 1
	*/
	a = (double *)malloc(sizeof(double) * MATRIX_I * MATRIX_J);
	b = (double *)malloc(sizeof(double) * MATRIX_J * MATRIX_K); 
	c1 = (double *)malloc(sizeof(double) * MATRIX_I * MATRIX_K);
	c2 = (double *)malloc(sizeof(double) * MATRIX_I * MATRIX_K);

	for(int i = 0; i < MATRIX_I * MATRIX_J; i++)
		a[i] = rand() % 100;
	for(int i = 0; i < MATRIX_J * MATRIX_K; i++)
		b[i] = rand() % 100;
	
	timer.initTimers();
	// end of init
 

	// serial version
	timer.onTimer(0);
	for(int i = 0; i < MATRIX_I; i++)
		for(int j = 0; j < MATRIX_J; j++)
			c1[i] += a[(i * MATRIX_J) + j] * b[j];
	timer.offTimer(0);

	// cuda version
        cudaMalloc((void **) &d_a, sizeof(double) * MATRIX_I * MATRIX_J);
	cudaMalloc((void **) &d_b, sizeof(double) * MATRIX_J * MATRIX_K);
	cudaMalloc((void **) &d_c, sizeof(double) * MATRIX_I * MATRIX_K);


	// send input data from host to device
	timer.onTimer(1);
	cudaMemcpy(d_a, a, sizeof(double) * MATRIX_I * MATRIX_J, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, b, sizeof(double) * MATRIX_J * MATRIX_K, cudaMemcpyHostToDevice);
	timer.offTimer(1);


	dim3 dimGrid(32, 1, 1);
	dim3 dimBlock(32, 1, 1);

	// kernel call
	timer.onTimer(2);
	matMul<<<dimGrid, dimBlock>>>(d_a, d_b, d_c);
	cudaThreadSynchronize();
	timer.offTimer(2);

	// send result from device to host
	timer.onTimer(3);
	cudaMemcpy(c2, d_c, sizeof(double) * MATRIX_I * MATRIX_K, cudaMemcpyDeviceToHost);
	timer.offTimer(3);

	
	// check sequence
	result = true;
	for(int i = 0; i < MATRIX_I; i++)
		if(c1[i] != c2[i])
		{
			printf("[%d] The results is not matched! (%lf, %lf)\n", i, c1[i], c2[i]);
			result = false;
		}
	
	if(result)
		printf("GPU works well!\n");
	

	timer.printTimer();


	cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);

	delete[] a; delete[] b; delete[] c1; delete[] c2;

	return 0;
}
