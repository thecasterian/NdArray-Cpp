#ifndef NDARRAY_FUNC_HPP
#define NDARRAY_FUNC_HPP

#include "ndarray-core.hpp"
#include "ndarray-slice.hpp"

namespace ndarray {

template <typename T>
NdArray<T, 1> arange(index_t stop) {
    return arange<T>(0, stop, 1);
}

template <typename T>
NdArray<T, 1> arange(T start, T stop) {
    return arange<T>(start, stop, 1);
}

template <typename T>
NdArray<T, 1> arange(index_t start, index_t stop, index_t step) {
    index_t size;
    if (step == 0) {
        throw std::invalid_argument("arange() step cannot be zero");
    }
    if (step > 0) {
        size = (stop - start + step - 1) / step;
    } else {
        size = (start - stop - step - 1) / (-step);
    }

    NdArray<T, 1> result(Shape({size}));
    for (index_t i = 0; i < size; ++i) {
        result[i] = start + i * step;
    }
    return result;
}

template <typename T, std::size_t Dim>
NdArray<T, Dim> ones(const Shape<Dim>& shape) {
    NdArray<T, Dim> result(shape);
    result.fill(1);
    return result;
}

template <typename T, std::size_t Dim>
NdArray<T, Dim> zeros(const Shape<Dim>& shape) {
    NdArray<T, Dim> result(shape);
    result.fill(0);
    return result;
}

}

#endif
