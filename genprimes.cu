/*
 * This file contains the code for doing the heat distribution problem.
 * You do not need to modify anything except starting  gpu_heat_dist() at the bottom
 * of this file.
 * In gpu_heat_dist() you can organize your data structure and the call to your
 * kernel(s) that you need to write too.
 *
 */

#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
 //#include <string.h>

 /* To index element (i,j) of a 2D array stored as 1D */
#define index(i, j, N) ((i) * (N)) + (j)
#define NUM_PER_THREAD 512

/*****************************************************************/

// Function declarations: Feel free to add any functions you want.
void gen_prime(bool*, unsigned long);
void printResult(float* playground, int N);

/*****************************************************************/
/**** Do NOT CHANGE ANYTHING in main() function ******/

int main(int argc, char* argv[]) {
    unsigned long N; /* Dimention of NxN matrix */

    /* The 2D array of points will be treated as 1D array of NxN elements */
    bool* playground;

    // to measure time taken by a specific part of the code
    // double time_taken;
    // clock_t start, end;

    if (argc != 2) {
        fprintf(stderr, "usage: genprimes N\n");
        exit(1);
    }

    // 49152 64 768 512

    N = (unsigned long)atol(argv[1]);

    /* Dynamically allocate NxN array of floats */
    playground = (bool*)calloc(N + 1, sizeof(bool));
    memset(playground, 1, (N + 1) * sizeof(bool));

    if (!playground) {
        fprintf(stderr, " Cannot allocate the %u x %u array\n", N, N);
        exit(1);
    }
    else  // The GPU version
    {
        // start = clock();
        gen_prime(playground, N);
        // end = clock();
    }

    // time_taken = ((double)(end - start)) / CLOCKS_PER_SEC;

    // printf("Time taken : %lf\n", time_taken);

    // printResult(playground, N);

    char file_name[20] = {};
    strcat(file_name, argv[1]);
    strcat(file_name, ".txt");
    printf("%s\n", file_name);

    FILE* fp;
    unsigned long count = 0;
    fp = fopen(file_name, "w+");
    for (unsigned long i = 2; i <= N; i++) {
        if (playground[i]) {
            count++;
            fprintf(fp, "%lu ", i);
        }
    }
    fclose(fp);

    free(playground);
    // printf("%lu", count);
    return 0;
}

void printResult(float* playground, int N) {
    printf("\n");
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            printf("%5.2f ", playground[index(i, j, N)]);
        }
        printf("\n");
    }
}

__global__ void simulate(bool* A, int N) {
    unsigned long t = blockDim.x * blockIdx.x + threadIdx.x;
    if (t < 2) return;
    unsigned long tmp = t;
    for (unsigned long i = 0; i <= (N - t) / t; i++) {
        tmp += t;
        A[tmp] = false;
    }
}

/***************** The GPU version: Write your code here *********************/
/* This function can call one or more kernels if you want ********************/
void gen_prime(bool* playground, unsigned long N) {
    bool* A_d;

    int blockDimSize = 64;  // 64

    unsigned long num_bytes = 0;
    unsigned long max = (unsigned long)ceil(sqrt(N));
    unsigned long num_to_process = max + blockDimSize - 1;
    num_bytes = (N + 1) * sizeof(bool);

    cudaMalloc((void**)&A_d, num_bytes);
    cudaMemcpy(A_d, playground, num_bytes, cudaMemcpyHostToDevice);

    unsigned long gridNum = (num_to_process) / blockDimSize;

    //dim3 dimGrid((N + blockDimSize - 1) / blockDimSize, (N + blockDimSize - 1) / blockDimSize);
    //dim3 dimBlock(blockDimSize, blockDimSize);

    simulate << <gridNum, blockDimSize >> > (A_d, N);

    cudaMemcpy(playground, A_d, num_bytes, cudaMemcpyDeviceToHost);
    cudaFree(A_d);

}
