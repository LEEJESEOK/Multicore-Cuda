#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "DS_timer.h"

#define N (1024 * 1024 * 128)

#define f(x) ((x) * (x))


__global__ trap_kernel(double a, double b, double h, int n, double * sum)
{
	const int NUM_THREAD_IN_BLOCK = blockDim.x * blockDim.y * blockDim.z;

	int bID = blockIdx.z * (gridDim.y * gridDim.x * NUM_THREAD_IN_BLOCK) + blockIdx.y * (gridDim.x * NUM_THREAD_IN_BLOCK) + (blockIdx.x * (blockDim.x * blockDim.y * blockDim.z));
	int tID = bID + ((blockDim.y * blockDim.x) * threadIdx.z) + (blockDim.x * threadIdx.y) + threadIdx.x;

	if(tId >= n - 1) return;

	atomicAdd(sum, d * h);
}


int main()
{
	double a, b, h, * dSum;

	DS_timer timer(2);
	timer.initTimers();

	// CPU version
	timer.onTimer(0);
	timer.offTimer(0);

	timer.printTimer();


}
