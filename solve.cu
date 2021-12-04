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
void calculate_x(int N, float precision, float* x, float* a, float* b, unsigned blockDimSize, unsigned* count);
void print_input(int N, float precision, float* x, float* a, float* b);
void printResult(float* playground, int N, unsigned count);

/*****************************************************************/
/**** Do NOT CHANGE ANYTHING in main() function ******/

int main(int argc, char* argv[]) {
    int N; /* Dimention of NxN matrix */
    float precision;

    /* The 2D array of points will be treated as 1D array of NxN elements */
    float* x_1;
    //float* x_2;
    float* b;
    float* a;

    // to measure time taken by a specific part of the code
    //double time_taken;
    //clock_t start, end;

    if (argc < 2) {
        fprintf(stderr, "usage: sovle <inputfile>\n");
        exit(1);
    }


    // read input ==========================================================

    int block_size = 64; // default block_size
    if (argc == 3) {
        block_size = atoi(argv[2]);
    }
    printf("#thread: %d\n", block_size);

    FILE* myFile;
    myFile = fopen(argv[1], "r");

    fscanf(myFile, "%d", &N);

    /* Dynamically allocate two array of floats for old */
    x_1 = (float*)calloc(N, sizeof(float));
    //x_2 = (float*)calloc(N , sizeof(float));
    b = (float*)calloc(N, sizeof(float));
    a = (float*)calloc(N * N, sizeof(float));

    memset(x_1, 0, (N) * sizeof(float));
    //memset(x_2, 0, (N ) * sizeof(float));
    memset(b, 0, (N) * sizeof(float));
    memset(a, 0, (N * N) * sizeof(float));

    fscanf(myFile, "%f", &precision);

    for (size_t i = 0; i < N; i++) {
        fscanf(myFile, "%f", &x_1[i]);
    }

    for (size_t i = 0; i < N; i++) {
        for (size_t j = 0; j < N; j++) {
            fscanf(myFile, "%f", &a[i * N + j]);
        }
        fscanf(myFile, "%f", &b[i]);
    }

    //print_input(N, precision, x_1, a, b);
    fclose(myFile);
    // read input end ==========================================================


    unsigned count;
    if (!a) {
        fprintf(stderr, " Cannot allocate the %u x %u array\n", N, N);
        exit(1);
    }
    else  // The GPU version
    {
        //start = clock();
        calculate_x(N, precision, x_1, a, b, block_size, &count); // todo blocksize
        //end = clock();
    }

    printResult(x_1, N, count);

    //time_taken = ((double)(end - start)) / CLOCKS_PER_SEC;

    //printf("Time taken : %lf\n", time_taken);

    free(a); 
    free(x_1); 
    free(b); 
    return 0;
}

void print_input(int N, float precision, float* x, float* a, float* b) {
    printf("%d\n", N);
    printf("%.5f\n", precision);
    for (size_t i = 0; i < N; i++) {
        printf("%.2f ", x[i]);
    }
    printf("\n");
    for (size_t i = 0; i < N; i++) {
        for (size_t j = 0; j < N; j++) {
            printf("%.2f ", a[i * N + j]);
        }
        printf("%.2f\n", b[i]);
    }
}

void printResult(float* x_1, int N, unsigned count) {
    /*printf("\n");
    for (size_t i = 0; i < N; i++)
    {
        printf("%.6f ", x_1[i]);
    }*/
    
    //printf("iterations: %d\n", count);
    printf("iterations: %u\n", count);

    // write result to file
    char file_name[30] = {};
    sprintf(file_name, "%d.sol", N);
    printf("%s\n", file_name);

    FILE* fp;
    fp = fopen(file_name, "w+");
    for (unsigned i = 0; i < N; i++) {
        fprintf(fp, "%f\n", x_1[i]);
    }
    fclose(fp);
}


__global__ void simulate(float precision, float* a, float* x_old, float* x_new, float* b, bool* flag_continue, int N) {

    unsigned long tid = blockDim.x * blockIdx.x + threadIdx.x;
    float sum = 0.0;
    if (tid < N) {
        //*flag_continue = false; // should be set to false in Host
        for (unsigned i = 0; i < N; i++)
        {
            if (i != tid) {
                sum += x_old[i] * (a[tid * N + i]);
            }
        }
        x_new[tid] = (b[tid] - sum) / a[tid * N + tid];
        if (fabsf((x_new[tid] - x_old[tid]) / x_new[tid]) > precision) {
            *flag_continue = true;
        }
    }
}


/***************** The GPU version: Write your code here *********************/
/* This function can call one or more kernels if you want ********************/
void calculate_x(int N, float precision, float* x, float* a, float* b, unsigned block_size, unsigned* count) {
    float* a_d;
    float* x_old_d;
    float* x_new_d;
    float* b_d;
    bool flag_continue = false;
    bool* flag_continue_d;


    unsigned num_bytes_a = N * N * sizeof(float);
    unsigned num_bytes_x = N * sizeof(float);
    unsigned num_bytes_flag = 1 * sizeof(bool);

    cudaMalloc((void**)&a_d, num_bytes_a);
    cudaMalloc((void**)&x_old_d, num_bytes_x);
    cudaMalloc((void**)&x_new_d, num_bytes_x);
    cudaMalloc((void**)&b_d, num_bytes_x);
    cudaMalloc((void**)&flag_continue_d, num_bytes_flag);

    cudaMemcpy(a_d, a, num_bytes_a, cudaMemcpyHostToDevice);
    cudaMemcpy(x_old_d, x, num_bytes_x, cudaMemcpyHostToDevice);
    cudaMemcpy(b_d, b, num_bytes_x, cudaMemcpyHostToDevice);
    

    unsigned  blockNum = (N + block_size - 1) / block_size;
        
    printf("#block: %d\n", blockNum);

    //dim3 dimGrid((N + blockDimSize - 1) / blockDimSize, (N + blockDimSize - 1) / blockDimSize);
    //dim3 dimBlock(blockDimSize, blockDimSize);


    *count = 0;
    while (true)
    {
        (*count)++;
        flag_continue = false;
        cudaMemcpy(flag_continue_d, &flag_continue, num_bytes_flag, cudaMemcpyHostToDevice);
        simulate <<<blockNum, block_size >>> (precision, a_d, x_old_d, x_new_d, b_d, flag_continue_d, N);
        cudaMemcpy(&flag_continue, flag_continue_d, num_bytes_flag, cudaMemcpyDeviceToHost);
        if (flag_continue) {
            float* tmp = x_old_d;
            x_old_d = x_new_d;
            x_new_d = tmp;
        }
        else {
            break;
        }
    }

    cudaMemcpy(x, x_new_d, num_bytes_x, cudaMemcpyDeviceToHost);

    cudaFree(a_d);
    cudaFree(x_old_d);
    cudaFree(x_new_d);
    cudaFree(b_d);
    cudaFree(flag_continue_d);

}
