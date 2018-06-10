#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "DS_timer.h"

#define f(x) ((x) * (x))


__global__ void trap_kernel(float a, float b, float h, int n, float * sum)
{
	int tID = blockIdx.x * blockDim.x + threadIdx.x;

	if(tID >= n - 1) return;
	
	float x_i = a + h * tID;
	float x_j = a + h * (tID + 1);
	float d = (f(x_i) + f(x_j)) / 2.0;

	atomicAdd(sum, d * h);
}

__global__ void trap_kernel_s1(float a, float b, float h, int n, float * sum)
{
	int tID = blockIdx.x * blockDim.x + threadIdx.x;

	if(tID >= n - 1) return;
	
	__shared__ float localSum;

	float x_i = a + h * tID;
	float x_j = a + h * (tID + 1);
	float d = (f(x_i) + f(x_j)) / 2.0;

	atomicAdd(&localSum, d * h);
	__syncthreads();

	if(threadIdx.x == 0) atomicAdd(sum, localSum);
}

__global__ void trap_kernel_s2(float a, float b, float h, int n, float * sum)
{
	int tID = blockIdx.x * blockDim.x + threadIdx.x;

	if(tID >= n - 1) return;
	
	__shared__ float localVal[64];
	localVal[blockDim.x] = 0;

	float x_i = a + h * tID;
	float x_j = a + h * (tID + 1);
	float d = (f(x_i) + f(x_j)) / 2.0;

	localVal[blockDim.x] = d * h;
	__syncthreads();

	if(threadIdx.x == 0)
	{
		for(int i = 1; i < blockDim.x; i++)
			localVal[0] += localVal[i];

		atomicAdd(sum, localVal[0]);
	}

}

__global__ void trap_kernel_s3(float a, float b, float h, int n, float * sum)
{
	int tID = blockIdx.x * blockDim.x + threadIdx.x;

	if(tID >= n - 1) return;
	
	__shared__ float localVal[64];
	localVal[blockDim.x] = 0;

	float x_i = a + h * tID;
	float x_j = a + h * (tID + 1);
	float d = (f(x_i) + f(x_j)) / 2.0;

	localVal[blockDim.x] = d * h;
	__syncthreads();

	// reduction
	int offset = 1;
	while(offset < blockDim.x)
	{
		if(threadIdx.x % (2 * offset) == 0)
			localVal[threadIdx.x] += localVal[threadIdx.x + offset];

		__syncthreads();
		offset *= 2;
	}

	if(threadIdx.x == 0)
		atomicAdd(sum, localVal[0]);
}

__global__ void trap_kernel_s4(float a, float b, float h, int n, float * sum)
{
	int tID = blockIdx.x * blockDim.x + threadIdx.x;

	if(tID >= n - 1) return;
	
	__shared__ float localVal[64];
	localVal[blockDim.x] = 0;

	float x_i = a + h * tID;
	float x_j = a + h * (tID + 1);
	float d = (f(x_i) + f(x_j)) / 2.0;

	localVal[blockDim.x] = d * h;
	__syncthreads();

	// reduction
	int offset = blockDim.x / 2;
	
	while(offset > 0)
	{
		if(threadIdx.x < offset)
			localVal[threadIdx.x] += localVal[threadIdx.x + offset];

		offset /= 2;

		__syncthreads();
	}

	if(threadIdx.x == 0)
		atomicAdd(sum, localVal[0]);


}

int main()
{
	float a, b, h;
	int n;
   	float sum = 0, * cuda_sum, * d_sum;

	DS_timer timer(6);
	timer.initTimers();

	printf("a > ");
	scanf("%f", &a);
	printf("b > ");
	scanf("%f", &b);
	printf("n > ");
	scanf("%d", &n);

	h = (b - a) / (float) n;

	// CPU version
	timer.setTimerName(0, (char *)"CPU");
	timer.onTimer(0);
	for(int i = 0; i < n - 1; i++)
	{
		float x_i = a + h * i;
		float x_j = a + h * (i + 1);
		float d = (f(x_i) + f(x_j)) / 2.0;
		sum += d * h;
	}
	timer.offTimer(0);
	printf("\tCPU sum : %f\n", sum);

	// CUDA version
	cuda_sum = (float *)malloc(sizeof(float));
	cudaMalloc((void **)&d_sum, sizeof(float));
	cudaMemset(d_sum, 0, sizeof(float));

	dim3 dimGrid(n / 64, 1, 1);
	dim3 dimBlock(64, 1, 1);

	// Global Sync
	timer.setTimerName(1, (char *)"Global Sync");
	
	timer.onTimer(1);
	trap_kernel<<<dimGrid, dimBlock>>>(a, b, h, n, d_sum);
	cudaThreadSynchronize();
	timer.offTimer(1);
	cudaMemcpy(cuda_sum, d_sum, sizeof(float), cudaMemcpyDeviceToHost);
	printf("\tCUDA sum : %f\n", *cuda_sum);	


	// Shared ver1
	cudaMemset(d_sum, 0, sizeof(float));
	
	timer.setTimerName(2, (char *)"Shared Ver1");
	timer.onTimer(2);
	trap_kernel_s1<<<dimGrid, dimBlock>>>(a, b, h, n, d_sum);
	cudaThreadSynchronize();
	timer.offTimer(2);
	cudaMemcpy(cuda_sum, d_sum, sizeof(float), cudaMemcpyDeviceToHost);
	printf("\tShared v1 sum : %f\n", *cuda_sum);	



	// Shared ver2
	cudaMemset(d_sum, 0, sizeof(float));
	
	timer.setTimerName(3, (char *)"Shared Ver2");
	timer.onTimer(3);
	trap_kernel_s2<<<dimGrid, dimBlock>>>(a, b, h, n, d_sum);
	cudaThreadSynchronize();
	timer.offTimer(3);
	cudaMemcpy(cuda_sum, d_sum, sizeof(float), cudaMemcpyDeviceToHost);
	printf("\tShared v2 sum : %f\n", *cuda_sum);	



	// Reduction 1
	cudaMemset(d_sum, 0, sizeof(float));
	
	timer.setTimerName(4, (char *)"Reduction Ver1");
	timer.onTimer(4);
	trap_kernel_s3<<<dimGrid, dimBlock>>>(a, b, h, n, d_sum);
	cudaThreadSynchronize();
	timer.offTimer(4);
	cudaMemcpy(cuda_sum, d_sum, sizeof(float), cudaMemcpyDeviceToHost);
	printf("\tReduction1 sum : %f\n", *cuda_sum);	



	// Reduction 2
	cudaMemset(d_sum, 0, sizeof(float));
	
	timer.setTimerName(5, (char *)"Reduction Ver2");
	timer.onTimer(5);
	trap_kernel_s4<<<dimGrid, dimBlock>>>(a, b, h, n, d_sum);
	cudaThreadSynchronize();
	timer.offTimer(5);
	cudaMemcpy(cuda_sum, d_sum, sizeof(float), cudaMemcpyDeviceToHost);
	printf("\tReduction2 sum : %f\n", *cuda_sum);	




	timer.printTimer();

	cudaFree(d_sum);
}
