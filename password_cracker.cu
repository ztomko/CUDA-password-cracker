#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime_api.h>

#define MAX_SIZE 6
#define VARIATION 93
#define FIRST 33
#define SPREAD 4

__constant__ char password[MAX_SIZE];

__device__ bool checkWord(char* guessword, int length){
  bool checkMatch = true;
  for(int i = 0; i <= length + 1; i++){
    if(guessword[i] != password[i]){
      checkMatch = false;
      break;
    }
  }
  return checkMatch;
}

__device__ void wordStarter(char* guessword, int length){
  guessword[0] = FIRST + threadIdx.x;
  if(length > 4) length = 4;
  switch(length){
    case 4: guessword[4] = FIRST + (blockIdx.y / VARIATION);
    case 3: guessword[3] = FIRST + (blockIdx.y % VARIATION);
    case 2: guessword[2] = FIRST + (blockIdx.x / VARIATION);
    case 1: guessword[1] = FIRST + (blockIdx.x % VARIATION);
  }
}

__global__ void startCrackin(char* testword){
  int length = 0;
  if(blockIdx.y > 0){
    if(blockIdx.y > VARIATION){
      length = 4;
    } else {
      length = 3;
    }
  } else if (blockIdx.x > 0){
    if(blockIdx.x > VARIATION){
      length = 2;
    } else {
      length = 1;
    }
  }
  char guessword[MAX_SIZE];
  guessword[length] = 0;
  wordStarter(guessword, length);
  if(checkWord(guessword, length)){
    for(int i = 0; i <= length + 1; i++){
      testword[i] = guessword[i];
    }
  }
}

int main(void){
  char* testword;
  char* hostPassword;
  char* cudaTestword;
  int starters = VARIATION * VARIATION;
  dim3 blocks = dim3(starters, starters, 1);
  cudaEvent_t start, stop;

  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  testword = (char*)malloc(MAX_SIZE * sizeof(char));
  hostPassword = (char*)malloc(MAX_SIZE * sizeof(char));
  cudaMalloc((void**) &cudaTestword, sizeof(char) * MAX_SIZE);
  cudaMemcpy(cudaTestword, testword, sizeof(char) * MAX_SIZE, cudaMemcpyHostToDevice);

  printf("Please enter a password:\n - cannot be over %d characters\n - can contain letters, numbers, and symbols\n - no spaces\n", MAX_SIZE - 1);
  scanf("%s", hostPassword);

  cudaMemcpyToSymbol(password, hostPassword, MAX_SIZE * sizeof(char));

  cudaEventRecord(start);
  startCrackin<<<blocks, VARIATION>>>(cudaTestword);
  cudaEventRecord(stop);

  cudaMemcpy(testword, cudaTestword, sizeof(char) * MAX_SIZE, cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();

  float runtime;
  cudaEventElapsedTime(&runtime, start, stop);
  printf("It took %.6f milliseconds to guess the word %s\n", runtime, testword);
  free(testword);
  free(hostPassword);
  cudaFree(cudaTestword);
}
