#pragma once
#ifndef _POINT_H_
#define _POINT_H_
#ifdef __cplusplus
extern "C++"{
#endif // __cplusplus
#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include <sstream>
#include <fstream>

class Point{
    public:
    int num = -1;
    // axis start and end
    float x_start = 0, x_end = 10;
    float y_start = 0, y_end = 10;
    float z_start = 0, z_end = 10;
    // tile num
    int xn = -1, yn = -1, zn = -1;
    // tile length
    float xl = 0, yl = 0, zl = 0;
    // grid length
    float gxl = 0, gyl = 0, gzl = 0;
    // position memory
    float *pos_cpu = nullptr, *pos_gpu = nullptr;
    // grid memory
    uint *index_cpu = nullptr, *index_gpu = nullptr;
    int *distance_cpu = nullptr, *distance_gpu = nullptr;
    uint *sum_cpu = nullptr, *sum_gpu = nullptr;
    uint *sum_out_cpu = nullptr, *sum_out_gpu = nullptr;
    uint *tile_index_with_point_cpu = nullptr, *tile_index_with_point_gpu = nullptr;
    uint *point_index_in_tile_cpu = nullptr, *point_index_in_tile_gpu = nullptr;
    uint *point_index_cpu = nullptr, *point_index_gpu = nullptr;
    uint grid_num, tile_num;
    Point();
    Point(std::string fileName);
    ~Point();
    int set_tile(int x_n = 16, int y_n = 16, int z_n = 16);
    int sample();
    int sample_2();
    int sample_3();
};

#ifdef __cplusplus
};
#endif // __cplusplus
#endif // _POINT_H_