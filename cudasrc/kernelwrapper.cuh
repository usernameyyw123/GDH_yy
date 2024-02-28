#ifndef STORAGE_LEVELDB_CUDASRC_KERNELWRAPPER_CUH_
#define STORAGE_LEVELDB_CUDASRC_KERNELWRAPPER_CUH_

#include <cuda.h>
#include <cuda_runtime.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <thrust/device_malloc.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>

#include <ctime>
#include <iostream>
#include <thread>

#define HANDLE_ERROR(err) (HandleError(err, __FILE__, __LINE__))

static void HandleError(cudaError_t err, const char *file, int line) {
  if (err != cudaSuccess) {
    fprintf(stderr, "Error %d: \"%s\" in %s at line %d\n", int(err),
            cudaGetErrorString(err), file, line);
    exit(int(err));
  }
}

// void cudaKernelFunc(std::vector<std::string>& hotvec);

void cudaKernelFunc();

struct sortstruct {
  char gpuslice[30 + 1];
  int gpuslicelen;
  __host__ __device__ sortstruct() {
    gpuslice[30] = '\0';
    gpuslicelen = 31;
  }

  constexpr __host__ __device__ bool operator==(sortstruct &strb) {
    char *ptra = (*this).gpuslice;
    char *ptrb = strb.gpuslice;
    for (int i = 0; i < 16; i++) {
      if ((*ptra) == (*ptrb)) {
        ptra++;
        ptrb++;
      } else {
        return false;
      }
    }
    return true;
  }
};

struct gpu_cmp {
  __host__ __device__ bool operator()(sortstruct &stra, sortstruct &strb) {
    char *ptra = stra.gpuslice;
    char *ptrb = strb.gpuslice;
    int count = 0;
    for (; (*ptra) == (*ptrb); ptra++, ptrb++) {
      if (*ptra == '\0') return 0;
      if (count == 15) return false;
      count++;
    }
    int ret = (*ptra) - (*ptrb);
    if (ret >= 0) return false;
    return true;
  }
};

struct gpu_equal {
  __host__ __device__ bool operator()(sortstruct &stra) {
    char const *ptrb = stra.gpuslice;
    char const *ptra = "0001000100010001";

    for (int i = 0; i < 16; i++) {
      if ((*ptra) == (*ptrb)) {
        ptra++;
        ptrb++;
      } else {
        return false;
      }
    }
    return true;
  }
};

// template <typename T = sortstruct>
// struct thrust::equal_to<sortstruct>
// {
// __thrust_exec_check_disable__ __host__ __device__ constexpr bool
// operator()(const sortstruct & lhs,const sortstruct & rhs) const{
//   const char *ptra = lhs.gpuslice;
//   const char *ptrb = rhs.gpuslice;
//   for(int i = 0 ; i < 16 ; i++){
//     if((*ptra) == (*ptrb)){
//       ptra++;
//       ptrb++;
//     }else{
//       return false;
//     }
//   }
//   return true;
// }
// };

// __host__ __device__ bool operator>(std::string &stra,std::string &strb){
//     const char* ptra = stra.c_str();
//         const char* ptrb = strb.c_str();
//         for(;*ptra==*ptra;ptra++,ptrb++)
//         if(*ptra==0)
//         return 0;
//         return (*ptra)-(*ptrb);
// }

#endif