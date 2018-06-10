#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "DS_timer.h"

#define DATA_SIZE (1024 * 1024 * 256)
#define DATA_RANGE (256)

void printHist(int * arr, char * str);

__global__ void histogram_atomic(float * a, int * histo, int n) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if(tid >= n) return;
	atomicAdd(histo + (int)a[tid], 1);
}

__global__ void histogram_shared(float * a, int * histo, int n) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	__shared__ int sh[DATA_RANGE];

	if(threadIdx.x < 256) sh[threadIdx.x] = 0;
	__syncthreads();
	
	if(tid < n) atomicAdd(&sh[(int)a[tid]], 1);
	__syncthreads();

	if(threadIdx.x < 256) atomicAdd(&histo[threadIdx.x], sh[threadIdx.x]);	

}


int main()
{
	float * arr, * d_arr;
	int * a, * b, * c;
	int * d_b, *d_c;
	
	DS_timer timer(3);


	arr = (float *) malloc(sizeof(float) * DATA_SIZE);

	a = (int *) malloc(sizeof(int) * DATA_RANGE);
	b = (int *) malloc(sizeof(int) * DATA_RANGE);
	c = (int *) malloc(sizeof(int) * DATA_RANGE);
	
	for(int i = 0; i < DATA_SIZE; i++)
		arr[i] = rand() % DATA_RANGE;
	for(int i = 0; i < DATA_RANGE; i++)
		a[i] = 0;

	timer.initTimers();


	// CPU version	
	timer.setTimerName(0, (char *)"CPU");
	timer.onTimer(0);
	for(int i = 0; i < DATA_SIZE; i++)
		a[(int) arr[i]]++;
	timer.offTimer(0);
	printHist(a, (char *)"Serial version");


	
	// Global Sync version
	cudaMalloc((void **)&d_arr, sizeof(float) * DATA_SIZE);
	cudaMalloc((void **)&d_b, sizeof(int) * DATA_RANGE);

	cudaMemcpy(d_arr, arr, sizeof(float) * DATA_SIZE, cudaMemcpyHostToDevice);

	cudaMemset(d_b, 0, sizeof(int) * DATA_RANGE);
	
	timer.setTimerName(1, (char *)"Global Sync");
	timer.onTimer(1);
	histogram_atomic<<<DATA_SIZE / 256, 256>>>(d_arr, d_b, DATA_SIZE);
	cudaThreadSynchronize();
	timer.offTimer(1);
	
	cudaMemcpy(b, d_b, sizeof(int) * DATA_RANGE, cudaMemcpyDeviceToHost);
	printHist(b, (char *) "Global Sync");


	// Shared Sync version
	cudaMalloc((void **)&d_c, sizeof(int) * DATA_RANGE);
	
	cudaMemset(d_c, 0, sizeof(int) * DATA_RANGE);
	
	timer.setTimerName(2, (char *)"Shared Sync");
	timer.onTimer(2);
	histogram_shared<<<DATA_SIZE / 256, 256>>>(d_arr, d_c, DATA_SIZE);
	cudaThreadSynchronize();
	timer.offTimer(2);
	
	cudaMemcpy(c, d_c, sizeof(int) * DATA_RANGE, cudaMemcpyDeviceToHost);
	printHist(c, (char *) "Shared Sync");
	

	timer.printTimer();


	free(arr);
	free(a); free(b);

	cudaFree(d_arr); cudaFree(d_b); cudaFree(d_c);

	return 0;
}


void printHist(int * arr, char * str)
{
	printf("\t<< %s >>\n", str);
	for(int i = 0; i < DATA_RANGE; i++)
	{
		printf(" <%3d : %d>", i, arr[i]);
		if(i % 8 == 7)
			printf("\n");
	}
	printf("\n");
}
