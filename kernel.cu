
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#define BLOCK_SIZE 32
__global__ void kernel_global(float *a, float *b, int n, float *c)
{
	int bx = blockIdx.x; // номер блока по x
	int by = blockIdx.y; // номер блока по y
	int tx = threadIdx.x; // номер нити в блоке по x
	int ty = threadIdx.y; // номер нити в блоке по y
	float sum = 0.0f;
	int ia = n * (BLOCK_SIZE * by + ty); // номер строки из AТ
	int ib = BLOCK_SIZE * bx + tx; // номер столбца из BТ
	int ic = ia + ib; // номер элемента из —Т
	// вычисление элемента матрицы C
	for (int k = 0; k < n; k++) sum += a[ia + k] * b[ib + k * n];
	c[ic] = sum;
}

int main()
{
	int N = 2048;
	int m, n, k;

	float timerValueGPU = 0.0f;
	float timerValueCPU = 0.0f;
	cudaEvent_t start;
	cudaEvent_t stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaError_t err;

	int numBytes = N * N * sizeof(float);
	float* adev, *bdev, *cdev, *a, *b, *c, *cc;

	a = (float*)malloc(numBytes);
	b = (float*)malloc(numBytes);
	c = (float*)malloc(numBytes);
	cc = (float*)malloc(numBytes);

	for (n = 0; n < N; n++)
		for (m = 0; m < N; m++) {
			a[m + n * N] = 2.0f * m + n;
			b[m + n * N] = 1;
		}

	dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
	dim3 blocks(N / threads.x, N / threads.y);

	cudaMalloc((void**)&adev, numBytes);
	cudaMalloc((void**)&bdev, numBytes);
	cudaMalloc((void**)&cdev, numBytes);



	cudaEventRecord(start, 0);

	cudaMemcpy(adev, a, numBytes, cudaMemcpyHostToDevice);
	cudaMemcpy(bdev, b, numBytes, cudaMemcpyHostToDevice);

	kernel_global << < blocks, threads >> > (adev, bdev, N, cdev);
	err = cudaPeekAtLastError();
	if (err != cudaSuccess)
		printf(cudaGetErrorString(err));
	cudaMemcpy(c, cdev, numBytes, cudaMemcpyDeviceToHost);
	cudaThreadSynchronize();
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&timerValueGPU, start, stop);
	printf("\n GPU calculation time %f msec\n", timerValueGPU);



	cudaEventRecord(start, 0);

	clock_t startc;
	clock_t stopc;

	startc = clock();

	for (n = 0; n < N; n++)
		for (m = 0; m < N; m++) {
			cc[m + n * N] = 0.f;
			for (k = 0; k < N; k++)
				cc[m + n * N] += a[k + n * N] * b[k + m * N];
		}

	stopc = clock();

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&timerValueCPU, start, stop);
	printf("\n CPU calculation time %f msec\n", timerValueCPU);
	printf("\n Ctime: %f\n", ((double)(stopc - startc)) / ((double)CLOCKS_PER_SEC));

	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	cudaFree(&adev);
	cudaFree(&bdev);
	cudaFree(&cdev);

	delete a;
	delete b;
	delete c;
	delete cc;

	return 0;
}