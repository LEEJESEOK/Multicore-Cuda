#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "DS_timer.h"

#define DATA_SIZE (1024 * 1024 * 512)
#define DATA_RANGE (256)

void printHist(int * arr, char * str);

int main()
{
	float * arr;
	int * a, * b;
	
	DS_timer timer(2);


	arr = (float *) malloc(sizeof(float) * DATA_SIZE);
	a = (int *) malloc(sizeof(int) * DATA_RANGE);
	b = (int *) malloc(sizeof(int) * DATA_RANGE);

	for(int i = 0; i < DATA_SIZE; i++)
		arr[i] = rand() % DATA_RANGE;
	for(int i = 0; i < DATA_RANGE; i++)
		a[i] = 0;


	timer.initTimers();
	
	timer.onTimer(0);
	for(int i = 0; i < DATA_SIZE; i++)
		a[(int) arr[i]]++;
	timer.offTimer(0);
	printHist(a, (char *)"serial version");

	timer.printTimer();

	free(arr);
	free(a); free(b);

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
