#ifdef __cplusplus
extern "C++"{
#endif // __cplusplus
#include "point.h"

Point::Point(){
    num = 0;
    std::cout<<"构造函数"<<std::endl;
}

Point::Point(std::string fileName){
    std::ifstream fid(fileName);
    num = 300000000;
    cudaHostAlloc((void**)&pos_cpu, num * 4 * sizeof(float), cudaHostAllocDefault);
    std::string line;
    float x;
    for(int i = 0; i < num; i++) {
        getline(fid, line);
        std::istringstream s(line);
        for(int j = 0; j < 3; j++) {
            s >> x;
            pos_cpu[j*num + i] = x;
        }
    }
    cudaMalloc((void**)&pos_gpu, num * 4 * sizeof(float));
    cudaMemcpy(pos_gpu, pos_cpu, num * 4 * sizeof(float), cudaMemcpyHostToDevice);
}

Point::~Point(){
    num = -1;
    cudaFreeHost(pos_cpu);
    cudaFree(pos_gpu);
    cudaFreeHost(index_cpu);
    cudaFree(index_gpu);
    cudaFreeHost(distance_cpu);
    cudaFree(distance_gpu);
    cudaFreeHost(sum_cpu);
    cudaFree(sum_gpu);
    cudaFreeHost(sum_out_cpu);
    cudaFree(sum_out_gpu);
    cudaFreeHost(tile_index_with_point_cpu);
    cudaFree(tile_index_with_point_gpu);
    cudaFreeHost(point_index_in_tile_cpu);
    cudaFree(point_index_in_tile_gpu);
    cudaFreeHost(point_index_cpu);
    cudaFree(point_index_gpu);
    std::cout<<"析构函数"<<std::endl;
}

__global__ void set_tile_kernel(){}

int Point::set_tile(int x_n, int y_n, int z_n){
    xn = x_n;
    yn = y_n;
    zn = z_n;
    grid_num = xn*yn*zn*23*23*23;
    tile_num = xn*yn*zn;
    xl = (x_end - x_start) / xn;
    yl = (y_end - y_start) / yn;
    zl = (z_end - z_start) / zn;
    gxl = (x_end - x_start) / (xn * 23);
    gyl = (y_end - y_start) / (yn * 23);
    gzl = (z_end - z_start) / (zn * 23);
    // set grid memory
    cudaHostAlloc((void**)&index_cpu, grid_num * sizeof(uint), cudaHostAllocDefault);
    cudaMalloc((void**)&index_gpu, grid_num * sizeof(uint));
    cudaHostAlloc((void**)&distance_cpu, grid_num * sizeof(int), cudaHostAllocDefault);
    cudaMalloc((void**)&distance_gpu, grid_num * sizeof(int));
    cudaHostAlloc((void**)&sum_cpu, 4096 * sizeof(uint), cudaHostAllocDefault);
    cudaMalloc((void**)&sum_gpu, 4096 * sizeof(uint));
    cudaHostAlloc((void**)&sum_out_cpu, 4096 * sizeof(uint), cudaHostAllocDefault);
    cudaMalloc((void**)&sum_out_gpu, 4096 * sizeof(uint));
    // 记录每个点，在每个tile的顺序
    cudaHostAlloc((void**)&tile_index_with_point_cpu, num * sizeof(uint), cudaHostAllocDefault);
    cudaMalloc((void**)&tile_index_with_point_gpu, num * sizeof(uint));
    cudaHostAlloc((void**)&point_index_in_tile_cpu, num * sizeof(uint), cudaHostAllocDefault);
    cudaMalloc((void**)&point_index_in_tile_gpu, num * sizeof(uint));
    cudaHostAlloc((void**)&point_index_cpu, num * sizeof(uint), cudaHostAllocDefault);
    cudaMalloc((void**)&point_index_gpu, num * sizeof(uint));
    for(int i = 0; i < grid_num; i++){
        distance_cpu[i] = 2e9;
    }
    cudaMemcpy(distance_gpu, distance_cpu, grid_num * sizeof(int), cudaMemcpyHostToDevice);
    for(int i = 0; i < 4096; i++){
        sum_cpu[i] = 0;
        //sum_out_cpu[i] = 1;
    }
    cudaMemcpy(sum_gpu, sum_cpu, 4096 * sizeof(uint), cudaMemcpyHostToDevice);
    //cudaMemcpy(sum_out_gpu, sum_out_cpu, 4096 * sizeof(uint), cudaMemcpyHostToDevice);
    
    return 1;
}

__global__ void sample_kernel(uint *index, int *distance, float *pos, int num,
                              float x_start, float y_start, float z_start,
                              float gxl, float gyl, float gzl){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    for(int i = tid; i < num; i += gridDim.x * blockDim.x){
        float x = (pos[i] - x_start) / gxl;
        float y = (pos[i+num] - y_start) / gyl;
        float z = (pos[i+2*num] - z_start) / gzl;
        uint x_index = uint(x);
        uint y_index = uint(y);
        uint z_index = uint(z);
        uint grid_index = x_index*360*360 + y_index*360 + z_index;
        float grid_distance = (x-x_index)*(x-x_index) + (y-y_index)*(y_index) + (z-z_index)*(z-z_index);
        int grid_distance_int = __float_as_int(grid_distance);
        atomicMin(&distance[grid_index], grid_distance_int);
        __syncthreads();
        if(grid_distance_int == distance[grid_index]){
            index[grid_index] = i;
        }
    }
}

int Point::sample(){
    sample_kernel<<<1024, 1024>>>(index_gpu, distance_gpu, pos_gpu, num,
                                  x_start, y_start, z_start, gxl, gyl, gzl);
    return 1;
}

__global__ void scan(uint *g_odata, uint *g_idata, int n){
	extern __shared__ uint temp[];  // allocated on invocation
	int thid = threadIdx.x;
	int offset = 1;
    uint first_value = 0, last_value;
    for(int i = 0; i < n; i += 2048){
    	temp[2*thid] = g_idata[2*thid + i];  // load input into shared memory
    	temp[2*thid+1] = g_idata[2*thid+1 + i];
    	if (thid == 0) {
            temp[0] += first_value;
        }
    	for (int d = 2048>>1; d > 0; d >>= 1){ 
    		__syncthreads();
    		if (thid < d){
    			int ai = offset*(2*thid+1)-1;
    			int bi = offset*(2*(thid+1))-1;
    			temp[bi] += temp[ai];
    		}
    		offset *= 2;
    	}
    	if (thid == 0) {
            last_value = temp[2048 - 1];
            temp[2048 - 1] = 0;
        }
    	for (int d = 1; d < 2048; d *= 2){
    		offset >>= 1;
    		__syncthreads();
    		if (thid < d){
    			int ai = offset*(2*thid+1)-1;
    			int bi = offset*(2*(thid+1))-1;
    			uint t = temp[ai];
    			temp[ai] = temp[bi];
    			temp[bi] += t; 
    		}
    	}
    	__syncthreads();
    	g_odata[2*thid + i] = temp[2*thid]; // write results to device memory
    	g_odata[2*thid+1 + i] = temp[2*thid+1];
    	if (thid == 0) {
            g_odata[i] = first_value;
            first_value = last_value;
        }
    }
}

__global__ void count_kernel(uint *index, int *distance, uint *sum_gpu,
                                uint *sum_out_gpu, uint *tile_index_with_point_gpu,
                                uint *point_index_in_tile_gpu, float *pos, int num,
                              float x_start, float y_start, float z_start,
                              float gxl, float gyl, float gzl){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    for(int i = tid; i < num; i += gridDim.x * blockDim.x){
        float x = (pos[i] - x_start) / (gxl*23);
        float y = (pos[i+num] - y_start) / (gyl*23);
        float z = (pos[i+2*num] - z_start) / (gzl*23);
        uint x_index = uint(x);
        uint y_index = uint(y);
        uint z_index = uint(z);
        uint tile_index = x_index*16*16 + y_index*16 + z_index;
        uint point_index_in_tile = atomicAdd(&sum_gpu[tile_index], 1);
        tile_index_with_point_gpu[i] = tile_index;
        point_index_in_tile_gpu[i] = point_index_in_tile;

    }
    //__syncthreads();
    //__threadfence();
    //scan<<<1, 1024, 2048 * 4, cudaStreamTailLaunch>>>(sum_out_gpu, sum_gpu, 8192);
    //if(blockIdx.x == 0){
    //    extern __shared__ uint temp[2048];  // allocated on invocation
    //    int n = 8192;
    //    int thid = threadIdx.x;
    //    int offset = 1;
    //    uint first_value = 0, last_value;
    //    for(int i2 = 0; i2 < n; i2 += 2048){
    //    //    temp[2*thid] = 0;  // load input into shared memory
    //        temp[2*thid] = sum_gpu[2*thid + i2];  // load input into shared memory
    //        temp[2*thid+1] = sum_gpu[2*thid+1 + i2];
    //        if (thid == 0) {
    //            temp[0] += first_value;
    //        }
    //        for (int d = 2048>>1; d > 0; d >>= 1){ 
    //            __syncthreads();
    //            if (thid < d){
    //                int ai = offset*(2*thid+1)-1;
    //                int bi = offset*(2*(thid+1))-1;
    //                temp[bi] += temp[ai];
    //            }
    //            offset *= 2;
    //        }
    //        if (thid == 0) {
    //            last_value = temp[2048 - 1];
    //            temp[2048 - 1] = 0;
    //        }
    //        for (int d = 1; d < 2048; d *= 2){
    //            offset >>= 1;
    //            __syncthreads();
    //            if (thid < d){
    //                int ai = offset*(2*thid+1)-1;
    //                int bi = offset*(2*(thid+1))-1;
    //                uint t = temp[ai];
    //                temp[ai] = temp[bi];
    //                temp[bi] += t; 
    //            }
    //        }
    //        __syncthreads();
    //        sum_out_gpu[2*thid + i2] = temp[2*thid]; // write results to device memory
    //        sum_out_gpu[2*thid+1 + i2] = temp[2*thid+1];
    //        if (thid == 0) {
    //            sum_out_gpu[i2] = first_value;
    //            first_value = last_value;
    //        }
    //    }
    //}
}

__global__ void sort_index_kernel(uint *point_index_gpu,
                                  float *pos_gpu,
                                  uint *tile_index_with_point_gpu,
                                  uint *point_index_in_tile_gpu,
                                  uint *sum_out_gpu, int num){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    for(int i = tid; i < num; i += gridDim.x * blockDim.x){
        point_index_gpu[sum_out_gpu[tile_index_with_point_gpu[i]] + point_index_in_tile_gpu[i]] = i;
        //pos_gpu[sum_out_gpu[tile_index_with_point_gpu[i]] + point_index_in_tile_gpu[i] + 3*num] = pos_gpu[i+2*num];
        //pos_gpu[sum_out_gpu[tile_index_with_point_gpu[i]] + point_index_in_tile_gpu[i] + 2*num] = pos_gpu[i+num];
        //pos_gpu[sum_out_gpu[tile_index_with_point_gpu[i]] + point_index_in_tile_gpu[i] + num] = pos_gpu[i];
    }

    for(int i = tid; i < num; i += gridDim.x * blockDim.x){
        uint index = point_index_gpu[i];
        pos_gpu[i + 3*num] = pos_gpu[index+2*num];
        pos_gpu[i + 2*num] = pos_gpu[index+1*num];
        pos_gpu[i + 1*num] = pos_gpu[index+0*num];
    }
    //for(int i = tid; i < num; i += gridDim.x * blockDim.x){
    //    pos_gpu[sum_out_gpu[tile_index_with_point_gpu[i]] + point_index_in_tile_gpu[i] + 2*num] = pos_gpu[i+num];
    //}
    //for(int i = tid; i < num; i += gridDim.x * blockDim.x){
    //    pos_gpu[sum_out_gpu[tile_index_with_point_gpu[i]] + point_index_in_tile_gpu[i] + num] = pos_gpu[i];
    //}
}

__global__ void sample_point_kernel(uint *index_gpu, uint *point_index_gpu,
                                    uint *tile_index_with_point_gpu,
                                    uint *point_index_in_tile_gpu,
                                    uint *sum_gpu, uint *sum_out_gpu,
                                    float *pos_gpu, int num,
                                    float x_start, float y_start, float z_start,
                                    float gxl, float gyl, float gzl){
    extern __shared__ uint distance_sm[];  // allocated on invocation
    for (uint i = blockIdx.x; i < 4096; i += gridDim.x){
        uint point_num_in_tile = sum_gpu[i];
        uint base_index = sum_out_gpu[i];
        for (uint j = threadIdx.x; j < 12167; j += blockDim.x) {
            distance_sm[j] = 2e9;
        }
        //__syncthreads();
        for (uint j = threadIdx.x; j < point_num_in_tile; j += blockDim.x){
            //uint index = point_index_gpu[base_index + j];
            uint index = base_index + j;
            float x = (pos_gpu[index] - x_start) / gxl;
            float y = (pos_gpu[index+num] - y_start) / gyl;
            float z = (pos_gpu[index+2*num] - z_start) / gzl;
            uint x_index = uint(x) % 23;
            uint y_index = uint(y) % 23;
            uint z_index = uint(z) % 23;
            uint grid_index = x_index*23*23 + y_index*23 + z_index;
            float grid_distance = (x-x_index)*(x-x_index) + (y-y_index)*(y_index) + (z-z_index)*(z-z_index);
            uint grid_distance_int = __float_as_uint(grid_distance);
            atomicMin(&distance_sm[grid_index], grid_distance_int);
            //tile_index_with_point_gpu[base_index + j] = grid_index;
            //point_index_in_tile_gpu[base_index + j] = grid_distance_int;
        //__syncthreads();
            if(grid_distance_int == distance_sm[grid_index]){
                index_gpu[i*12167 + grid_index] = index;
            }
        }
        //__syncthreads();
        //for (uint j = threadIdx.x; j < point_num_in_tile; j += blockDim.x){
        //    uint index = point_index_gpu[base_index + j];
        //    uint grid_index = tile_index_with_point_gpu[base_index + j];
        //    uint grid_distance_int = point_index_in_tile_gpu[base_index + j];
        //    if(grid_distance_int == distance_sm[grid_index]){
        //        index_gpu[i*5832 + grid_index] = index;
        //    }
        //}
    }
}

int Point::sample_2(){
    count_kernel<<<1024, 1024>>>(index_gpu, distance_gpu, sum_gpu, sum_out_gpu,
                                            tile_index_with_point_gpu, point_index_in_tile_gpu, pos_gpu, num,
                                  x_start, y_start, z_start, gxl, gyl, gzl);
    scan<<<1, 1024, 2048 * sizeof(uint)>>>(sum_out_gpu, sum_gpu, 4096);
    sort_index_kernel<<<1024, 1024>>>(point_index_gpu, pos_gpu, tile_index_with_point_gpu, point_index_in_tile_gpu, sum_out_gpu, num);
    //scan<<<1, 1024, 2048 * 4, cudaStreamTailLaunch>>>(sum_out_gpu, sum_gpu, 8192);
    sample_point_kernel<<<1024, 1024, 12167 * sizeof(uint)>>>(index_gpu, point_index_gpu, tile_index_with_point_gpu,
                                       point_index_in_tile_gpu, sum_gpu, sum_out_gpu,
                                       pos_gpu+num, num, x_start, y_start, z_start,
                                       gxl, gyl, gzl);
    cudaMemcpy(sum_out_cpu, sum_out_gpu, 4096 * sizeof(uint), cudaMemcpyDeviceToHost);
    for(int i = 0; i < 4096; i++){
        std::cout<<i<<":"<<sum_out_cpu[i]<<" ";
    }
    std::cout<<std::endl;
    cudaMemcpy(point_index_cpu, point_index_gpu, 4096 * sizeof(uint), cudaMemcpyDeviceToHost);
    for(int i = 0; i < 4096; i++){
        std::cout<<i<<":"<<point_index_cpu[i]<<" ";
    }
    std::cout<<std::endl;
    return 1;
}

//__global__ void sample_3_kernel(uint *index, int *distance, uint *sum_gpu, uint *sum_out_gpu, float *pos, int num,
//                                float x_start, float y_start, float z_start,
//                                float gxl, float gyl, float gzl){
//	extern __shared__ uint temp[2048];  // allocated on invocation
//    int tid = blockIdx.x * blockDim.x + threadIdx.x;
//    for(int i = tid; i < num; i += gridDim.x * blockDim.x){
//        float x = (pos[i] - x_start) / (gxl*18);
//        float y = (pos[i+num] - y_start) / (gyl*18);
//        float z = (pos[i+2*num] - z_start) / (gzl*18);
//        uint x_index = uint(x);
//        uint y_index = uint(y);
//        uint z_index = uint(z);
//        uint tile_index = x_index*20*20 + y_index*20 + z_index;
//        uint point_index_in_tile = atomicAdd(&sum_gpu[tile_index], 1);
//        __syncthreads();
//}

int Point::sample_3(){
    return 1;
}

#ifdef __cplusplus
};
#endif // __cplusplus