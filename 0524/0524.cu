#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "DS_timer.h"


void printMat(float *mat, int size_i, int isze_j);

//__global__ void matMul(float *a, float *b, float *c, int size_i, int size_j, int size_k)
__global__ void matMul(float *a, float *b, float *c, int M, int N, int K)
{
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;

	if(row > M || col > N)
		return;
	
	float sum = 0.f;

	__syncthreads();

	for(int k = 0; k < K; k++)
	{
		sum += a[row * K + k] * b[k * N + col];
	}

	__syncthreads();

	c[row * N + col] = sum;
}

int main()
{
	// definition
	float *a, *b, *c, *d;
	float *d_a, *d_b, *d_c;

	int size_i, size_j, size_k;

	DS_timer timer(4);

	// input size

	printf("i > ");
	scanf("%d", &size_i); 
	printf("j > ");
	scanf("%d", &size_j);
	printf("k > ");
	scanf("%d", &size_k); 

	// init
	/*
	a : size_i by size_j
	b : size_j by size_k
	c, d : size_i by size_k
	*/
	a = (float *)malloc(sizeof(float) * size_i * size_j);
	b = (float *)malloc(sizeof(float) * size_j * size_k);
        c = (float *)malloc(sizeof(float) * size_i * size_k);
        d = (float *)malloc(sizeof(float) * size_i * size_k);
	
	// random vlaue (0.0000 ~ 9.9999)
	for(int i = 0; i < size_i * size_j; i++)
                a[i] = (rand() % 10) / 10.f;
	for(int i = 0; i < size_j * size_k; i++)
                b[i] = (rand() % 10) / 10.f;

	timer.initTimers();
	// end of init


	// serial version
	timer.onTimer(0);
/*
	for(int i = 0; i < size_i; i++)
		for(int j = 0; j < size_j; j++)
			for(int k  = 0; k < size_k; k++)
				c[(i * size_k) + k] += a[(i * size_j) + j] * b[(j * size_k) + k];
*/				
	for(int i = 0; i < size_i; i++)
	{
		for(int k = 0; k < size_k; k++)
		{
			float sum = 0.f;
			for(int j = 0; j < size_j; j++)
				sum += a[i * size_j + j] * b[j * size_k + k];
			c[i * size_k + k] = sum;
		}
	}
	timer.offTimer(0);	
	
	// cuda version
        cudaMalloc((void **) &d_a, sizeof(float) * size_i * size_j);
        cudaMalloc((void **) &d_b, sizeof(float) * size_j * size_k);
        cudaMalloc((void **) &d_c, sizeof(float) * size_i * size_k);

	// send input data from host to device
	timer.onTimer(1);
        cudaMemcpy(d_a, a, sizeof(float) * size_i * size_j, cudaMemcpyHostToDevice);
        cudaMemcpy(d_b, b, sizeof(float) * size_j * size_k, cudaMemcpyHostToDevice);
	timer.offTimer(1);

	// 
	dim3 gridDim((size_i + 8 - 1) / 8, (size_k + 8 - 1) / 8, 1);
//	dim3 gridDim(size_i / 8, size_k / 8, 1);
	dim3 blockDim(8, 8, 1);
	// kernel call
	timer.onTimer(2);
	matMul<<<gridDim, blockDim>>>(d_a, d_b, d_c, size_i, size_k, size_j);
	cudaThreadSynchronize();
	timer.offTimer(2);

	// send result from device to host
	timer.onTimer(3);
	cudaMemcpy(d, d_c, sizeof(float) * size_i * size_k, cudaMemcpyDeviceToHost);
	timer.offTimer(3);


	// check sequence
	bool result = true;

	for(int i = 0; i < size_i * size_k; i++)
	{
		if(c[i] != d[i])
		{
			printf("[%d] The results is not matched! (%f, %f)\n", i, c[i], d[i]);
			result = false;
		}
	}

	if(result)
		printf("GPU works well!\n");
	
	timer.printTimer();


        cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
        delete[] a; delete[] b; delete[] c, delete[] d;

	return 0;
}

void printMat(float *mat, int size_i, int size_j)
{
	for(int i = 0; i < size_i; i++)
	{
		for(int j = 0; j < size_j; j++)
			printf("%f ", mat[(i * size_j) + j]);
		printf("\n");
	}
}
