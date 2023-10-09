#include <iostream>
#include "point.h"
#include <math.h>

int main(void){
    std::cout<<"this is point program"<<std::endl;
    Point p = Point("point.txt");
    p.set_tile();
    //p.sample();
    p.sample_2();
    std::cout<<p.xn<<std::endl;
    std::cout<<p.xl<<std::endl;
    std::cout<<p.num<<std::endl;
    return 0;
}