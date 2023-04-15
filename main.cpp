#include <iostream>

#include "ndarray.hpp"

int main(void) {
    ndarray::NdArray<int, 1> a = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};

    std::cout << a[0] << ' ' << a[1] << std::endl;

    auto sa = a[ndarray::Slice(2, 5)];

    std::cout << sa[0] << ' ' << sa[1] << std::endl;

    ndarray::NdArray<int, 3> c = {{{0, 1, 2}, {3, 4, 5}, {6, 7, 8}},
                                  {{9, 10, 11}, {12, 13, 14}, {15, 16, 17}},
                                  {{18, 19, 20}, {21, 22, 23}, {24, 25, 26}}};

    return 0;
}
