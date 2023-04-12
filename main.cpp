#include <iostream>

#include "ndarray.hpp"

int main(void) {
    nda::NdArray<int, 3> a = {
        {
            {1, 2, 3},
            {4, 5, 6},
        },
        {
            {7, 8, 9},
            {10, 11, 12},
        },
    };

    std::cout << a[0, 0, 0] << std::endl;
    std::cout << a[1, 1, 2] << std::endl;
    std::cout << a[-1, -2, -1] << std::endl;

    return 0;
}
