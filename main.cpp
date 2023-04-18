#include <iostream>

#include "ndarray.hpp"

int main(void) {
    ndarray::NdArray<int, 1> a = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    const ndarray::NdArray<int, 1> b = {42, 42, 42};

    std::cout << a[0] << std::endl;
    std::cout << b[0] << std::endl;

    const ndarray::NdArray<int, 3> c = {{{0, 1, 2}, {3, 4, 5}, {6, 7, 8}},
                                  {{9, 10, 11}, {12, 13, 14}, {15, 16, 17}},
                                  {{18, 19, 20}, {21, 22, 23}, {24, 25, 26}}};

    std::cout << c[0, 0, 0] << ' ' << c[1, 1, 1] << ' ' << c[2, 2, 2] << std::endl;

    std::cout << c[0][ndarray::Slice(2)].size() << std::endl;

    return 0;
}
