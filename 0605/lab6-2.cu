#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "DS_timer.h"

#define f(x) ((x) * (x))


__global__ void trap_kernel(double a, double b, double h, int n, double * sum)
{
	const int NUM_THREAD_IN_BLOCK = blockDim.x * blockDim.y * blockDim.z;

	int bID = blockIdx.z * (gridDim.y * gridDim.x * NUM_THREAD_IN_BLOCK) + blockIdx.y * (gridDim.x * NUM_THREAD_IN_BLOCK) + (blockIdx.x * (blockDim.x * blockDim.y * blockDim.z));
	int tID = bID + ((blockDim.y * blockDim.x) * threadIdx.z) + (blockDim.x * threadIdx.y) + threadIdx.x;

	if(tID >= n - 1) return;


//	atomicAdd(sum, d * h);
}


int main()
{
	double a, b, h;
	int n;
   	double sum = 0, * d_sum;

	DS_timer timer(2);
	timer.initTimers();

	printf("a > ");
	scanf("%d", &a);
	printf("b > ");
	scanf("%d", &b);
	printf("n > ");
	scanf("%d", &n);

	h = (b - a) / (double) N;

	// CPU version
	timer.onTimer(0);
	for(int i = 0; i < N - 1; i++)
	{
		double x_i = a + h * i;
		double x_j = a + h * (i + 1);
		double d = (f(x_i) + f(x_j)) / 2.0;
		sum += d * h;
	}
	timer.offTimer(0);
	printf("\tCPU sum : %lf\n", sum);

	// CUDA version

	cudaMalloc((void **)&d_sum, sizeof(double));

	dim3 dimGrid(N / 64, 1, 1);
	dim3 dimBlock(64, 1, 1);
	timer.onTimer(1);
	trap_kernel<<<dimGrid, dimBlock>>>(a, b, h, n, &d_sum);
	timer.offTimer(1);

	timer.printTimer();

	cudaFree(d_sum);
}
