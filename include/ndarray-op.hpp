#ifndef NDARRAY_OP_HPP
#define NDARRAY_OP_HPP

#include <cmath>

#include "ndarray-base.hpp"
#include "ndarray-core.hpp"

namespace ndarray {

/* Comparison operators ***********************************************************************************************/

template <typename T, std::size_t Dim, typename Derived1, typename Derived2>
const NdArray<bool, Dim> operator==(const NdArrayBase<T, Dim, Derived1> &lhs,
                                    const NdArrayBase<T, Dim, Derived2> &rhs) {
    util::validate_shape_binary_op(lhs.shape, rhs.shape);

    NdArray<bool, Dim> result(lhs.shape);
    std::size_t size = lhs.shape.size();
    for (std::size_t i = 0; i < size; ++i) {
        result.item(i) = lhs.item(i) == rhs.item(i);
    }
    return result;
}

template <typename T, std::size_t Dim, typename Derived1, typename Derived2>
const NdArray<bool, Dim> operator!=(const NdArrayBase<T, Dim, Derived1> &lhs,
                                    const NdArrayBase<T, Dim, Derived2> &rhs) {
    util::validate_shape_binary_op(lhs.shape, rhs.shape);

    NdArray<bool, Dim> result(lhs.shape);
    std::size_t size = lhs.shape.size();
    for (std::size_t i = 0; i < size; ++i) {
        result.item(i) = lhs.item(i) != rhs.item(i);
    }
    return result;
}

template <typename T, std::size_t Dim, typename Derived1, typename Derived2>
const NdArray<bool, Dim> operator<(const NdArrayBase<T, Dim, Derived1> &lhs, const NdArrayBase<T, Dim, Derived2> &rhs) {
    util::validate_shape_binary_op(lhs.shape, rhs.shape);

    NdArray<bool, Dim> result(lhs.shape);
    std::size_t size = lhs.shape.size();
    for (std::size_t i = 0; i < size; ++i) {
        result.item(i) = lhs.item(i) < rhs.item(i);
    }
    return result;
}

template <typename T, std::size_t Dim, typename Derived1, typename Derived2>
const NdArray<bool, Dim> operator>(const NdArrayBase<T, Dim, Derived1> &lhs, const NdArrayBase<T, Dim, Derived2> &rhs) {
    util::validate_shape_binary_op(lhs.shape, rhs.shape);

    NdArray<bool, Dim> result(lhs.shape);
    std::size_t size = lhs.shape.size();
    for (std::size_t i = 0; i < size; ++i) {
        result.item(i) = lhs.item(i) > rhs.item(i);
    }
    return result;
}

template <typename T, std::size_t Dim, typename Derived1, typename Derived2>
const NdArray<bool, Dim> operator<=(const NdArrayBase<T, Dim, Derived1> &lhs,
                                    const NdArrayBase<T, Dim, Derived2> &rhs) {
    util::validate_shape_binary_op(lhs.shape, rhs.shape);

    NdArray<bool, Dim> result(lhs.shape);
    std::size_t size = lhs.shape.size();
    for (std::size_t i = 0; i < size; ++i) {
        result.item(i) = lhs.item(i) <= rhs.item(i);
    }
    return result;
}

template <typename T, std::size_t Dim, typename Derived1, typename Derived2>
const NdArray<bool, Dim> operator>=(const NdArrayBase<T, Dim, Derived1> &lhs,
                                    const NdArrayBase<T, Dim, Derived2> &rhs) {
    util::validate_shape_binary_op(lhs.shape, rhs.shape);

    NdArray<bool, Dim> result(lhs.shape);
    std::size_t size = lhs.shape.size();
    for (std::size_t i = 0; i < size; ++i) {
        result.item(i) = lhs.item(i) >= rhs.item(i);
    }
    return result;
}

/* Binary arithmetic operators ****************************************************************************************/

template <typename T, std::size_t Dim, typename Derived1, typename Derived2>
const NdArray<T, Dim> operator+(const NdArrayBase<T, Dim, Derived1> &lhs, const NdArrayBase<T, Dim, Derived2> &rhs) {
    util::validate_shape_binary_op(lhs.shape, rhs.shape);

    NdArray<T, Dim> result(lhs.shape);
    std::size_t size = lhs.shape.size();
    for (std::size_t i = 0; i < size; ++i) {
        result.item(i) = lhs.item(i) + rhs.item(i);
    }
    return result;
}

template <typename T, std::size_t Dim, typename Derived1, typename Derived2>
const NdArray<T, Dim> operator-(const NdArrayBase<T, Dim, Derived1> &lhs, const NdArrayBase<T, Dim, Derived2> &rhs) {
    util::validate_shape_binary_op(lhs.shape, rhs.shape);

    NdArray<T, Dim> result(lhs.shape);
    std::size_t size = lhs.shape.size();
    for (std::size_t i = 0; i < size; ++i) {
        result.item(i) = lhs.item(i) - rhs.item(i);
    }
    return result;
}

template <typename T, std::size_t Dim, typename Derived1, typename Derived2>
const NdArray<T, Dim> operator*(const NdArrayBase<T, Dim, Derived1> &lhs, const NdArrayBase<T, Dim, Derived2> &rhs) {
    util::validate_shape_binary_op(lhs.shape, rhs.shape);

    NdArray<T, Dim> result(lhs.shape);
    std::size_t size = lhs.shape.size();
    for (std::size_t i = 0; i < size; ++i) {
        result.item(i) = lhs.item(i) * rhs.item(i);
    }
    return result;
}

template <typename T, std::size_t Dim, typename Derived1, typename Derived2>
const NdArray<T, Dim> operator/(const NdArrayBase<T, Dim, Derived1> &lhs, const NdArrayBase<T, Dim, Derived2> &rhs) {
    util::validate_shape_binary_op(lhs.shape, rhs.shape);

    NdArray<T, Dim> result(lhs.shape);
    std::size_t size = lhs.shape.size();
    for (std::size_t i = 0; i < size; ++i) {
        result.item(i) = lhs.item(i) / rhs.item(i);
    }
    return result;
}

template <typename T, std::size_t Dim, typename Derived1, typename Derived2>
const NdArray<T, Dim> operator%(const NdArrayBase<T, Dim, Derived1> &lhs, const NdArrayBase<T, Dim, Derived2> &rhs) {
    util::validate_shape_binary_op(lhs.shape, rhs.shape);

    NdArray<T, Dim> result(lhs.shape);
    std::size_t size = lhs.shape.size();
    for (std::size_t i = 0; i < size; ++i) {
        if constexpr (std::is_floating_point_v<T>) {
            result.item(i) = std::fmod(lhs.item(i), rhs.item(i));
        } else {
            result.item(i) = lhs.item(i) % rhs.item(i);
        }
    }
    return result;
}

}  // namespace ndarray

#endif
