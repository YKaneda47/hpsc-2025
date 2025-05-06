#include <cstdio>
#include <cstdlib>
#include <vector>

__global__ void init_bucket(int* bucket, int range) {
    int i = threadIdx.x;
    if (i < range) {
        bucket[i] = 0;
    }
}

__global__ void count_keys(int* key, int* bucket, int n) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) {
        atomicAdd(&bucket[key[i]], 1);
    }
}

__global__ void sort(int* key, int* bucket, int range) {
    int i = threadIdx.x;
    if (i < range) {
        int count = bucket[i];
        int idx = 0;
        for (int j = 0; j < i; ++j) {
            idx += bucket[j];
        }
        for (int j = 0; j < count; ++j) {
            key[idx + j] = i;
        }
    }
}

int main() {
  int n = 50;
  int range = 5;
  //std::vector<int> key(n);
  int *key;
  int *bucket;
  cudaMallocManaged(&key, n*sizeof(int));
  cudaMallocManaged(&bucket, range*sizeof(int));

  for (int i=0; i<n; i++) {
    key[i] = rand() % range;
    printf("%d ",key[i]);
  }
  printf("\n");
/*
  std::vector<int> bucket(range); 
  for (int i=0; i<range; i++) {
    bucket[i] = 0;
  }
  for (int i=0; i<n; i++) {
    bucket[key[i]]++;
  }
  for (int i=0, j=0; i<range; i++) {
    for (; bucket[i]>0; bucket[i]--) {
      key[j++] = i;
    }
  }
*/
  init_bucket<<<1, range>>>(bucket, range);
  cudaDeviceSynchronize();

  count_keys<<<(n+255) / 256, 256>>>(key, bucket, n);
  cudaDeviceSynchronize();

  sort<<<1, range>>>(key, bucket, range);
  cudaDeviceSynchronize();

  for (int i=0; i<n; i++) {
    printf("%d ",key[i]);
  }
  printf("\n");

  cudaFree(key);
  cudaFree(bucket);
}
