#include "kernelwrapper.cuh"

__global__ void firstkernel() {
  // printf("hello,gpu\n");
}

void cudaKernelFunc() {}

// void cudaKernelFunc(std::vector<std::string>& hotvec){
//     // printf("hot size %zu .................\n",hotvec.size());
//     thrust::host_vector<sortstruct> htmp;

//     int control = 0;

//     // std::cout << "src hot vector :" << hotvec.size() << std::endl;
//     // for(auto iter : hotvec){
//     //     if(control == 5) break;
//     //     control++;
//     //     std::cout << iter.data() << std::endl;
//     // }

//     // thrust::sort(hotvec.begin(),hotvec.end(),gpu_cmp());
//     // auto h_iter = hotvec.begin();
//     for(auto iter : hotvec){
//         sortstruct stmp;
//         memset(stmp.gpuslice,0,stmp.gpuslicelen);
//         memcpy(stmp.gpuslice,iter.data(),iter.size());
//         htmp.push_back(stmp);
//     }

//     control = 0;
//     // std::cout << "hostvector temp :" << htmp.size() << std::endl;
//     // for(auto iter : htmp){
//     //     if(control == 5) break;
//     //     control++;
//     //     std::cout << iter.gpuslice << std::endl;
//     // }
//     sortstruct st;
//     std::string str("0001000100010001");
//     memcpy(st.gpuslice,str.c_str(),str.size());
//     st.gpuslicelen = str.size();
//     htmp.push_back(st);

//     thrust::device_vector<sortstruct> devvec = htmp;
//     thrust::sort(devvec.begin(),devvec.end(),gpu_cmp());

//     // htmp = devvec;
//     thrust::copy(devvec.begin(), devvec.end(), htmp.begin());
//     // std::cout << "sorted" << std::endl;

//     // control = 0;
//     // std::cout << "sorted hostvector temp :" << htmp.size() << std::endl;
//     // for(auto iter : htmp){
//     //     if(control == 5) break;
//     //     control++;
//     //     std::cout << iter.gpuslice << std::endl;
//     // }
//     // printf("\n");
//     firstkernel<<<1,2>>>();

//     //构建缓存空间

//     //首先获取显存总大小
//     cudaDeviceProp prop;
//     int count;
//     size_t totalGlobalMem;
//     float rate = 0.05;
//     cudaGetDeviceCount(&count); // 获取设备数目，比如NVIDIA GeForce GTX TITAN
//     X 有两个GPU（也就是双核） count为2 for (int i = 0; i < count; i++)
//     {
//         cudaGetDeviceProperties(&prop, i); //
//         将第i个GPU数据放到prop中，本设备为1
//         // std::cout << "显卡名称：" << prop.name << std::endl;
//         // std::cout << "显存大小：" << prop.totalGlobalMem << " MB" <<
//         std::endl; totalGlobalMem = prop.totalGlobalMem;
//         // std::cout << "一个block的共享内存大小：" << prop.sharedMemPerBlock
//         << " KB" << std::endl;
//         // std::cout << "block最大线程数：" << prop.maxThreadsPerBlock <<
//         std::endl;
//     }
//     totalGlobalMem = rate * totalGlobalMem;

//     // std::cout << "缓存空间最大值：" << totalGlobalMem << std::endl;
//     // char* gpukeyvaluebuffer = nullptr;
//     bool isgpumallocbuffer = false;
//     if(!isgpumallocbuffer){
// 		//malloc gpu memory
// 		// HANDLE_ERROR( cudaMalloc((void**)&gpukeyvaluebuffer,
// totalGlobalMem * sizeof(char)) );

//         //malloc thrust data struct memory
//         thrust::device_ptr<sortstruct> dev_ptr =
//         thrust::device_malloc<sortstruct>(10);

//         // thrust::equal_to<sortstruct> eqfun = gpu_equal();
//         thrust::device_vector<sortstruct>::iterator iter;
//         iter =
//         thrust::find_if(thrust::device,devvec.begin(),devvec.end(),gpu_equal());
//         thrust::host_vector<sortstruct> hostptr(1);
//         // sortstruct *sptr = thrust::raw_pointer_cast(&iter[0]);
//         // thrust::device_ptr<sortstruct> dptr =
//         thrust::device_pointer_cast(sptr);
//         // cudaMemcpy(sptr);

//         /////在GPU缓存中查找某个key
//         // if(iter == devvec.end())
//         // {
//         //     std::cout << "not find" << std::endl;
//         // }else{
//         //     thrust::copy(iter,iter+1,hostptr.begin());
//         //     std::cout << hostptr[0].gpuslice << std::endl;
//         // }

//         //////////////
//         thrust::find_if(devvec.begin(),devvec.end(),gpu_equal());
//     }

//     // cudaDeviceReset();
//     cudaDeviceSynchronize();

// };