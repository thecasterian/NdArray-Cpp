#include <iostream>

#include "ndarray.hpp"

int main(void) {
    ndarray::NdArray<int, 1> a = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    ndarray::NdArray<int, 1> b = {-1, -1, -1};

    a[ndarray::Slice(2, 5)] = b;

    return 0;
}
