#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "DS_timer.h"

#define NUM_DATA (512 * 512 * 256)

__global__ void matAdd(int *a, int *b, int *c)
{
	const int NUM_THREAD_IN_BLOCK = blockDim.x * blockDim.y * blockDim.z;

 	int bID = blockIdx.z * (gridDim.y * gridDim.x * NUM_THREAD_IN_BLOCK) + blockIdx.y * (gridDim.x * NUM_THREAD_IN_BLOCK) + (blockIdx.x * (blockDim.x * blockDim.y * blockDim.z));
	int tID = bID + ((blockDim.y * blockDim.x) * threadIdx.z) + (blockDim.x * threadIdx.y) + threadIdx.x;
	
	c[tID] = a[tID] + b[tID];
}

int main()
{
	// definition
	int *a, *b, *c, *d;
	int *d_a, *d_b, *d_c;

	bool result;

	DS_timer timer(4);


	// init
	a = (int *)malloc(sizeof(int) * NUM_DATA);
	b = (int *)malloc(sizeof(int) * NUM_DATA);
        c = (int *)malloc(sizeof(int) * NUM_DATA);
        d = (int *)malloc(sizeof(int) * NUM_DATA);

	for(int i = 0; i < NUM_DATA; i++)
	{
                a[i] = rand() % 10;
                b[i] = rand() % 10;
	}
	
	timer.initTimers();
	// end of init


	// serial version
	timer.onTimer(0);
	for(int i = 0; i < NUM_DATA; i++)
		c[i] = a[i] + b[i];
	timer.offTimer(0);	


	// cuda version
        cudaMalloc((void **) &d_a, sizeof(int) * NUM_DATA);
        cudaMalloc((void **) &d_b, sizeof(int) * NUM_DATA);
        cudaMalloc((void **) &d_c, sizeof(int) * NUM_DATA);

	// send input data from host to device
	timer.onTimer(1);
        cudaMemcpy(d_a, a, sizeof(int) * NUM_DATA, cudaMemcpyHostToDevice);
        cudaMemcpy(d_b, b, sizeof(int) * NUM_DATA, cudaMemcpyHostToDevice);
	timer.offTimer(1);

	// kernel call
	timer.onTimer(2);
	matAdd<<<NUM_DATA / 1024, 1024>>>(d_a, d_b, d_c);
	cudaThreadSynchronize();
	timer.offTimer(2);

	// send result from device to host
	timer.onTimer(3);
	cudaMemcpy(d, d_c, sizeof(int) * NUM_DATA, cudaMemcpyDeviceToHost);
	timer.offTimer(3);


	// check sequence
	result = true;
	for(int i = 0; i < NUM_DATA; i++)
	{
		if(c[i] != d[i])
		{
			printf("[%d] The results is not matchhed! (%d, %d)\n", i, c[i], d[i]);
			result = false;
		}
	} 

	if(result)
		printf("GPU works well!\n");

	
	timer.printTimer();


        cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
        delete[] a; delete[] b; delete[] c; delete[] d;

	return 0;
}
