#ifndef NDARRAY_OP_HPP
#define NDARRAY_OP_HPP

#include "ndarray-core.hpp"
#include "ndarray-slice.hpp"

namespace ndarray {

/* Comparison operators ***********************************************************************************************/

template <typename T, std::size_t Dim>
const NdArray<bool, Dim> operator==(const NdArray<T, Dim> &lhs, const NdArray<T, Dim> &rhs) {
    if (lhs.shape != rhs.shape)
        throw std::invalid_argument("Cannot compare two arrays with different shapes " + lhs.shape.to_string() +
                                    " and " + rhs.shape.to_string());

    NdArray<bool, Dim> result(lhs.shape);
    for (std::size_t i = 0; i < lhs.shape.size(); ++i) {
        result._data[i] = lhs._data[i] == rhs._data[i];
    }
    return result;
}

template <typename T, std::size_t Dim>
const NdArray<bool, Dim> operator!=(const NdArray<T, Dim> &lhs, const NdArray<T, Dim> &rhs) {
    if (lhs.shape != rhs.shape)
        throw std::invalid_argument("Cannot compare two arrays with different shapes " + lhs.shape.to_string() +
                                    " and " + rhs.shape.to_string());

    NdArray<bool, Dim> result(lhs.shape);
    for (std::size_t i = 0; i < lhs.shape.size(); ++i) {
        result._data[i] = lhs._data[i] != rhs._data[i];
    }
    return result;
}

template <typename T, std::size_t Dim>
const NdArray<bool, Dim> operator<(const NdArray<T, Dim> &lhs, const NdArray<T, Dim> &rhs) {
    if (lhs.shape != rhs.shape)
        throw std::invalid_argument("Cannot compare two arrays with different shapes " + lhs.shape.to_string() +
                                    " and " + rhs.shape.to_string());

    NdArray<bool, Dim> result(lhs.shape);
    for (std::size_t i = 0; i < lhs.shape.size(); ++i) {
        result._data[i] = lhs._data[i] < rhs._data[i];
    }
    return result;
}

template <typename T, std::size_t Dim>
const NdArray<bool, Dim> operator>(const NdArray<T, Dim> &lhs, const NdArray<T, Dim> &rhs) {
    if (lhs.shape != rhs.shape)
        throw std::invalid_argument("Cannot compare two arrays with different shapes " + lhs.shape.to_string() +
                                    " and " + rhs.shape.to_string());

    NdArray<bool, Dim> result(lhs.shape);
    for (std::size_t i = 0; i < lhs.shape.size(); ++i) {
        result._data[i] = lhs._data[i] > rhs._data[i];
    }
    return result;
}

template <typename T, std::size_t Dim>
const NdArray<bool, Dim> operator<=(const NdArray<T, Dim> &lhs, const NdArray<T, Dim> &rhs) {
    if (lhs.shape != rhs.shape)
        throw std::invalid_argument("Cannot compare two arrays with different shapes " + lhs.shape.to_string() +
                                    " and " + rhs.shape.to_string());

    NdArray<bool, Dim> result(lhs.shape);
    for (std::size_t i = 0; i < lhs.shape.size(); ++i) {
        result._data[i] = lhs._data[i] <= rhs._data[i];
    }
    return result;
}

template <typename T, std::size_t Dim>
const NdArray<bool, Dim> operator>=(const NdArray<T, Dim> &lhs, const NdArray<T, Dim> &rhs) {
    if (lhs.shape != rhs.shape)
        throw std::invalid_argument("Cannot compare two arrays with different shapes " + lhs.shape.to_string() +
                                    " and " + rhs.shape.to_string());

    NdArray<bool, Dim> result(lhs.shape);
    for (std::size_t i = 0; i < lhs.shape.size(); ++i) {
        result._data[i] = lhs._data[i] >= rhs._data[i];
    }
    return result;
}

}  // namespace ndarray

#endif
