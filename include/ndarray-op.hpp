#ifndef NDARRAY_OP_HPP
#define NDARRAY_OP_HPP

#include <cmath>

#include "ndarray-base.hpp"
#include "ndarray-core.hpp"

namespace ndarray {

/* Unary operators ****************************************************************************************************/

template <typename T, std::size_t Dim, typename Derived>
const NdArray<T, Dim> operator+(const NdArrayBase<T, Dim, Derived> &arr) {
    NdArray<T, Dim> result(arr.shape());
    std::size_t size = arr.shape().size();
    for (std::size_t i = 0; i < size; ++i) {
        result.item(i) = +arr.item(i);
    }
    return result;
}

template <typename T, std::size_t Dim, typename Derived>
const NdArray<T, Dim> operator-(const NdArrayBase<T, Dim, Derived> &arr) {
    NdArray<T, Dim> result(arr.shape());
    std::size_t size = arr.shape().size();
    for (std::size_t i = 0; i < size; ++i) {
        result.item(i) = -arr.item(i);
    }
    return result;
}

template <typename T, std::size_t Dim, typename Derived>
const NdArray<bool, Dim> operator!(const NdArrayBase<T, Dim, Derived> &arr) {
    NdArray<bool, Dim> result(arr.shape());
    std::size_t size = arr.shape().size();
    for (std::size_t i = 0; i < size; ++i) {
        result.item(i) = !arr.item(i);
    }
    return result;
}

/* Comparison operators ***********************************************************************************************/

template <typename T, std::size_t Dim, typename Derived1, typename Derived2>
const NdArray<bool, Dim> operator==(const NdArrayBase<T, Dim, Derived1> &lhs,
                                    const NdArrayBase<T, Dim, Derived2> &rhs) {
    util::validate_shape_binary_op(lhs.shape(), rhs.shape());

    NdArray<bool, Dim> result(lhs.shape());
    std::size_t size = lhs.shape().size();
    for (std::size_t i = 0; i < size; ++i) {
        result.item(i) = lhs.item(i) == rhs.item(i);
    }
    return result;
}

template <typename T, std::size_t Dim, typename Derived>
const NdArray<bool, Dim> operator==(const T &lhs, const NdArrayBase<T, Dim, Derived> &rhs) {
    NdArray<bool, Dim> result(rhs.shape());
    std::size_t size = rhs.shape().size();
    for (std::size_t i = 0; i < size; ++i) {
        result.item(i) = lhs == rhs.item(i);
    }
    return result;
}

template <typename T, std::size_t Dim, typename Derived>
const NdArray<bool, Dim> operator==(const NdArrayBase<T, Dim, Derived> &lhs, const T &rhs) {
    NdArray<bool, Dim> result(lhs.shape());
    std::size_t size = lhs.shape().size();
    for (std::size_t i = 0; i < size; ++i) {
        result.item(i) = lhs.item(i) == rhs;
    }
    return result;
}

template <typename T, std::size_t Dim, typename Derived1, typename Derived2>
const NdArray<bool, Dim> operator!=(const NdArrayBase<T, Dim, Derived1> &lhs,
                                    const NdArrayBase<T, Dim, Derived2> &rhs) {
    util::validate_shape_binary_op(lhs.shape(), rhs.shape());

    NdArray<bool, Dim> result(lhs.shape());
    std::size_t size = lhs.shape().size();
    for (std::size_t i = 0; i < size; ++i) {
        result.item(i) = lhs.item(i) != rhs.item(i);
    }
    return result;
}

template <typename T, std::size_t Dim, typename Derived>
const NdArray<bool, Dim> operator!=(const T &lhs, const NdArrayBase<T, Dim, Derived> &rhs) {
    NdArray<bool, Dim> result(rhs.shape());
    std::size_t size = rhs.shape().size();
    for (std::size_t i = 0; i < size; ++i) {
        result.item(i) = lhs != rhs.item(i);
    }
    return result;
}

template <typename T, std::size_t Dim, typename Derived>
const NdArray<bool, Dim> operator!=(const NdArrayBase<T, Dim, Derived> &lhs, const T &rhs) {
    NdArray<bool, Dim> result(lhs.shape());
    std::size_t size = lhs.shape().size();
    for (std::size_t i = 0; i < size; ++i) {
        result.item(i) = lhs.item(i) != rhs;
    }
    return result;
}

template <typename T, std::size_t Dim, typename Derived1, typename Derived2>
const NdArray<bool, Dim> operator<(const NdArrayBase<T, Dim, Derived1> &lhs, const NdArrayBase<T, Dim, Derived2> &rhs) {
    util::validate_shape_binary_op(lhs.shape(), rhs.shape());

    NdArray<bool, Dim> result(lhs.shape());
    std::size_t size = lhs.shape().size();
    for (std::size_t i = 0; i < size; ++i) {
        result.item(i) = lhs.item(i) < rhs.item(i);
    }
    return result;
}

template <typename T, std::size_t Dim, typename Derived>
const NdArray<bool, Dim> operator<(const T &lhs, const NdArrayBase<T, Dim, Derived> &rhs) {
    NdArray<bool, Dim> result(rhs.shape());
    std::size_t size = rhs.shape().size();
    for (std::size_t i = 0; i < size; ++i) {
        result.item(i) = lhs < rhs.item(i);
    }
    return result;
}

template <typename T, std::size_t Dim, typename Derived>
const NdArray<bool, Dim> operator<(const NdArrayBase<T, Dim, Derived> &lhs, const T &rhs) {
    NdArray<bool, Dim> result(lhs.shape());
    std::size_t size = lhs.shape().size();
    for (std::size_t i = 0; i < size; ++i) {
        result.item(i) = lhs.item(i) < rhs;
    }
    return result;
}

template <typename T, std::size_t Dim, typename Derived1, typename Derived2>
const NdArray<bool, Dim> operator>(const NdArrayBase<T, Dim, Derived1> &lhs, const NdArrayBase<T, Dim, Derived2> &rhs) {
    util::validate_shape_binary_op(lhs.shape(), rhs.shape());

    NdArray<bool, Dim> result(lhs.shape());
    std::size_t size = lhs.shape().size();
    for (std::size_t i = 0; i < size; ++i) {
        result.item(i) = lhs.item(i) > rhs.item(i);
    }
    return result;
}

template <typename T, std::size_t Dim, typename Derived>
const NdArray<bool, Dim> operator>(const T &lhs, const NdArrayBase<T, Dim, Derived> &rhs) {
    NdArray<bool, Dim> result(rhs.shape());
    std::size_t size = rhs.shape().size();
    for (std::size_t i = 0; i < size; ++i) {
        result.item(i) = lhs > rhs.item(i);
    }
    return result;
}

template <typename T, std::size_t Dim, typename Derived>
const NdArray<bool, Dim> operator>(const NdArrayBase<T, Dim, Derived> &lhs, const T &rhs) {
    NdArray<bool, Dim> result(lhs.shape());
    std::size_t size = lhs.shape().size();
    for (std::size_t i = 0; i < size; ++i) {
        result.item(i) = lhs.item(i) > rhs;
    }
    return result;
}

template <typename T, std::size_t Dim, typename Derived1, typename Derived2>
const NdArray<bool, Dim> operator<=(const NdArrayBase<T, Dim, Derived1> &lhs,
                                    const NdArrayBase<T, Dim, Derived2> &rhs) {
    util::validate_shape_binary_op(lhs.shape(), rhs.shape());

    NdArray<bool, Dim> result(lhs.shape());
    std::size_t size = lhs.shape().size();
    for (std::size_t i = 0; i < size; ++i) {
        result.item(i) = lhs.item(i) <= rhs.item(i);
    }
    return result;
}

template <typename T, std::size_t Dim, typename Derived>
const NdArray<bool, Dim> operator<=(const T &lhs, const NdArrayBase<T, Dim, Derived> &rhs) {
    NdArray<bool, Dim> result(rhs.shape());
    std::size_t size = rhs.shape().size();
    for (std::size_t i = 0; i < size; ++i) {
        result.item(i) = lhs <= rhs.item(i);
    }
    return result;
}

template <typename T, std::size_t Dim, typename Derived>
const NdArray<bool, Dim> operator<=(const NdArrayBase<T, Dim, Derived> &lhs, const T &rhs) {
    NdArray<bool, Dim> result(lhs.shape());
    std::size_t size = lhs.shape().size();
    for (std::size_t i = 0; i < size; ++i) {
        result.item(i) = lhs.item(i) <= rhs;
    }
    return result;
}

template <typename T, std::size_t Dim, typename Derived1, typename Derived2>
const NdArray<bool, Dim> operator>=(const NdArrayBase<T, Dim, Derived1> &lhs,
                                    const NdArrayBase<T, Dim, Derived2> &rhs) {
    util::validate_shape_binary_op(lhs.shape(), rhs.shape());

    NdArray<bool, Dim> result(lhs.shape());
    std::size_t size = lhs.shape().size();
    for (std::size_t i = 0; i < size; ++i) {
        result.item(i) = lhs.item(i) >= rhs.item(i);
    }
    return result;
}

template <typename T, std::size_t Dim, typename Derived>
const NdArray<bool, Dim> operator>=(const T &lhs, const NdArrayBase<T, Dim, Derived> &rhs) {
    NdArray<bool, Dim> result(rhs.shape());
    std::size_t size = rhs.shape().size();
    for (std::size_t i = 0; i < size; ++i) {
        result.item(i) = lhs >= rhs.item(i);
    }
    return result;
}

template <typename T, std::size_t Dim, typename Derived>
const NdArray<bool, Dim> operator>=(const NdArrayBase<T, Dim, Derived> &lhs, const T &rhs) {
    NdArray<bool, Dim> result(lhs.shape());
    std::size_t size = lhs.shape().size();
    for (std::size_t i = 0; i < size; ++i) {
        result.item(i) = lhs.item(i) >= rhs;
    }
    return result;
}

/* Binary arithmetic operators ****************************************************************************************/

template <typename T, std::size_t Dim, typename Derived1, typename Derived2>
const NdArray<T, Dim> operator+(const NdArrayBase<T, Dim, Derived1> &lhs, const NdArrayBase<T, Dim, Derived2> &rhs) {
    util::validate_shape_binary_op(lhs.shape(), rhs.shape());

    NdArray<T, Dim> result(lhs.shape());
    std::size_t size = lhs.shape().size();
    for (std::size_t i = 0; i < size; ++i) {
        result.item(i) = lhs.item(i) + rhs.item(i);
    }
    return result;
}

template <typename T, std::size_t Dim, typename Derived>
const NdArray<T, Dim> operator+(const T &lhs, const NdArrayBase<T, Dim, Derived> &rhs) {
    NdArray<T, Dim> result(rhs.shape());
    std::size_t size = rhs.shape().size();
    for (std::size_t i = 0; i < size; ++i) {
        result.item(i) = lhs + rhs.item(i);
    }
    return result;
}

template <typename T, std::size_t Dim, typename Derived>
const NdArray<T, Dim> operator+(const NdArrayBase<T, Dim, Derived> &lhs, const T &rhs) {
    NdArray<T, Dim> result(lhs.shape());
    std::size_t size = lhs.shape().size();
    for (std::size_t i = 0; i < size; ++i) {
        result.item(i) = lhs.item(i) + rhs;
    }
    return result;
}

template <typename T, std::size_t Dim, typename Derived1, typename Derived2>
const NdArray<T, Dim> operator-(const NdArrayBase<T, Dim, Derived1> &lhs, const NdArrayBase<T, Dim, Derived2> &rhs) {
    util::validate_shape_binary_op(lhs.shape(), rhs.shape());

    NdArray<T, Dim> result(lhs.shape());
    std::size_t size = lhs.shape().size();
    for (std::size_t i = 0; i < size; ++i) {
        result.item(i) = lhs.item(i) - rhs.item(i);
    }
    return result;
}

template <typename T, std::size_t Dim, typename Derived>
const NdArray<T, Dim> operator-(const T &lhs, const NdArrayBase<T, Dim, Derived> &rhs) {
    NdArray<T, Dim> result(rhs.shape());
    std::size_t size = rhs.shape().size();
    for (std::size_t i = 0; i < size; ++i) {
        result.item(i) = lhs - rhs.item(i);
    }
    return result;
}

template <typename T, std::size_t Dim, typename Derived>
const NdArray<T, Dim> operator-(const NdArrayBase<T, Dim, Derived> &lhs, const T &rhs) {
    NdArray<T, Dim> result(lhs.shape());
    std::size_t size = lhs.shape().size();
    for (std::size_t i = 0; i < size; ++i) {
        result.item(i) = lhs.item(i) - rhs;
    }
    return result;
}

template <typename T, std::size_t Dim, typename Derived1, typename Derived2>
const NdArray<T, Dim> operator*(const NdArrayBase<T, Dim, Derived1> &lhs, const NdArrayBase<T, Dim, Derived2> &rhs) {
    util::validate_shape_binary_op(lhs.shape(), rhs.shape());

    NdArray<T, Dim> result(lhs.shape());
    std::size_t size = lhs.shape().size();
    for (std::size_t i = 0; i < size; ++i) {
        result.item(i) = lhs.item(i) * rhs.item(i);
    }
    return result;
}

template <typename T, std::size_t Dim, typename Derived>
const NdArray<T, Dim> operator*(const T &lhs, const NdArrayBase<T, Dim, Derived> &rhs) {
    NdArray<T, Dim> result(rhs.shape());
    std::size_t size = rhs.shape().size();
    for (std::size_t i = 0; i < size; ++i) {
        result.item(i) = lhs * rhs.item(i);
    }
    return result;
}

template <typename T, std::size_t Dim, typename Derived>
const NdArray<T, Dim> operator*(const NdArrayBase<T, Dim, Derived> &lhs, const T &rhs) {
    NdArray<T, Dim> result(lhs.shape());
    std::size_t size = lhs.shape().size();
    for (std::size_t i = 0; i < size; ++i) {
        result.item(i) = lhs.item(i) * rhs;
    }
    return result;
}

template <typename T, std::size_t Dim, typename Derived1, typename Derived2>
const NdArray<T, Dim> operator/(const NdArrayBase<T, Dim, Derived1> &lhs, const NdArrayBase<T, Dim, Derived2> &rhs) {
    util::validate_shape_binary_op(lhs.shape(), rhs.shape());

    NdArray<T, Dim> result(lhs.shape());
    std::size_t size = lhs.shape().size();
    for (std::size_t i = 0; i < size; ++i) {
        result.item(i) = lhs.item(i) / rhs.item(i);
    }
    return result;
}

template <typename T, std::size_t Dim, typename Derived>
const NdArray<T, Dim> operator/(const T &lhs, const NdArrayBase<T, Dim, Derived> &rhs) {
    NdArray<T, Dim> result(rhs.shape());
    std::size_t size = rhs.shape().size();
    for (std::size_t i = 0; i < size; ++i) {
        result.item(i) = lhs / rhs.item(i);
    }
    return result;
}

template <typename T, std::size_t Dim, typename Derived>
const NdArray<T, Dim> operator/(const NdArrayBase<T, Dim, Derived> &lhs, const T &rhs) {
    NdArray<T, Dim> result(lhs.shape());
    std::size_t size = lhs.shape().size();
    for (std::size_t i = 0; i < size; ++i) {
        result.item(i) = lhs.item(i) / rhs;
    }
    return result;
}

template <typename T, std::size_t Dim, typename Derived1, typename Derived2>
const NdArray<T, Dim> operator%(const NdArrayBase<T, Dim, Derived1> &lhs, const NdArrayBase<T, Dim, Derived2> &rhs) {
    util::validate_shape_binary_op(lhs.shape(), rhs.shape());

    NdArray<T, Dim> result(lhs.shape());
    std::size_t size = lhs.shape().size();
    for (std::size_t i = 0; i < size; ++i) {
        if constexpr (std::is_floating_point_v<T>) {
            result.item(i) = std::fmod(lhs.item(i), rhs.item(i));
        } else {
            result.item(i) = lhs.item(i) % rhs.item(i);
        }
    }
    return result;
}

template <typename T, std::size_t Dim, typename Derived>
const NdArray<T, Dim> operator%(const T &lhs, const NdArrayBase<T, Dim, Derived> &rhs) {
    NdArray<T, Dim> result(rhs.shape());
    std::size_t size = rhs.shape().size();
    for (std::size_t i = 0; i < size; ++i) {
        if constexpr (std::is_floating_point_v<T>) {
            result.item(i) = std::fmod(lhs, rhs.item(i));
        } else {
            result.item(i) = lhs % rhs.item(i);
        }
    }
    return result;
}

template <typename T, std::size_t Dim, typename Derived>
const NdArray<T, Dim> operator%(const NdArrayBase<T, Dim, Derived> &lhs, const T &rhs) {
    NdArray<T, Dim> result(lhs.shape());
    std::size_t size = lhs.shape().size();
    for (std::size_t i = 0; i < size; ++i) {
        if constexpr (std::is_floating_point_v<T>) {
            result.item(i) = std::fmod(lhs.item(i), rhs);
        } else {
            result.item(i) = lhs.item(i) % rhs;
        }
    }
    return result;
}

template <typename T, std::size_t Dim, typename Derived1, typename Derived2>
const NdArray<T, Dim> operator<<(const NdArrayBase<T, Dim, Derived1> &lhs, const NdArrayBase<T, Dim, Derived2> &rhs) {
    util::validate_shape_binary_op(lhs.shape(), rhs.shape());

    NdArray<T, Dim> result(lhs.shape());
    std::size_t size = lhs.shape().size();
    for (std::size_t i = 0; i < size; ++i) {
        result.item(i) = lhs.item(i) << rhs.item(i);
    }
    return result;
}

template <typename T, std::size_t Dim, typename Derived>
const NdArray<T, Dim> operator<<(const T &lhs, const NdArrayBase<T, Dim, Derived> &rhs) {
    NdArray<T, Dim> result(rhs.shape());
    std::size_t size = rhs.shape().size();
    for (std::size_t i = 0; i < size; ++i) {
        result.item(i) = lhs << rhs.item(i);
    }
    return result;
}

template <typename T, std::size_t Dim, typename Derived>
const NdArray<T, Dim> operator<<(const NdArrayBase<T, Dim, Derived> &lhs, const T &rhs) {
    NdArray<T, Dim> result(lhs.shape());
    std::size_t size = lhs.shape().size();
    for (std::size_t i = 0; i < size; ++i) {
        result.item(i) = lhs.item(i) << rhs;
    }
    return result;
}

template <typename T, std::size_t Dim, typename Derived1, typename Derived2>
const NdArray<T, Dim> operator>>(const NdArrayBase<T, Dim, Derived1> &lhs, const NdArrayBase<T, Dim, Derived2> &rhs) {
    util::validate_shape_binary_op(lhs.shape(), rhs.shape());

    NdArray<T, Dim> result(lhs.shape());
    std::size_t size = lhs.shape().size();
    for (std::size_t i = 0; i < size; ++i) {
        result.item(i) = lhs.item(i) >> rhs.item(i);
    }
    return result;
}

template <typename T, std::size_t Dim, typename Derived>
const NdArray<T, Dim> operator>>(const T &lhs, const NdArrayBase<T, Dim, Derived> &rhs) {
    NdArray<T, Dim> result(rhs.shape());
    std::size_t size = rhs.shape().size();
    for (std::size_t i = 0; i < size; ++i) {
        result.item(i) = lhs >> rhs.item(i);
    }
    return result;
}

template <typename T, std::size_t Dim, typename Derived>
const NdArray<T, Dim> operator>>(const NdArrayBase<T, Dim, Derived> &lhs, const T &rhs) {
    NdArray<T, Dim> result(lhs.shape());
    std::size_t size = lhs.shape().size();
    for (std::size_t i = 0; i < size; ++i) {
        result.item(i) = lhs.item(i) >> rhs;
    }
    return result;
}

template <typename T, std::size_t Dim, typename Derived1, typename Derived2>
const NdArray<T, Dim> operator&(const NdArrayBase<T, Dim, Derived1> &lhs, const NdArrayBase<T, Dim, Derived2> &rhs) {
    util::validate_shape_binary_op(lhs.shape(), rhs.shape());

    NdArray<T, Dim> result(lhs.shape());
    std::size_t size = lhs.shape().size();
    for (std::size_t i = 0; i < size; ++i) {
        result.item(i) = lhs.item(i) & rhs.item(i);
    }
    return result;
}

template <typename T, std::size_t Dim, typename Derived>
const NdArray<T, Dim> operator&(const T &lhs, const NdArrayBase<T, Dim, Derived> &rhs) {
    NdArray<T, Dim> result(rhs.shape());
    std::size_t size = rhs.shape().size();
    for (std::size_t i = 0; i < size; ++i) {
        result.item(i) = lhs & rhs.item(i);
    }
    return result;
}

template <typename T, std::size_t Dim, typename Derived>
const NdArray<T, Dim> operator&(const NdArrayBase<T, Dim, Derived> &lhs, const T &rhs) {
    NdArray<T, Dim> result(lhs.shape());
    std::size_t size = lhs.shape().size();
    for (std::size_t i = 0; i < size; ++i) {
        result.item(i) = lhs.item(i) & rhs;
    }
    return result;
}

template <typename T, std::size_t Dim, typename Derived1, typename Derived2>
const NdArray<T, Dim> operator^(const NdArrayBase<T, Dim, Derived1> &lhs, const NdArrayBase<T, Dim, Derived2> &rhs) {
    util::validate_shape_binary_op(lhs.shape(), rhs.shape());

    NdArray<T, Dim> result(lhs.shape());
    std::size_t size = lhs.shape().size();
    for (std::size_t i = 0; i < size; ++i) {
        result.item(i) = lhs.item(i) ^ rhs.item(i);
    }
    return result;
}

template <typename T, std::size_t Dim, typename Derived>
const NdArray<T, Dim> operator^(const T &lhs, const NdArrayBase<T, Dim, Derived> &rhs) {
    NdArray<T, Dim> result(rhs.shape());
    std::size_t size = rhs.shape().size();
    for (std::size_t i = 0; i < size; ++i) {
        result.item(i) = lhs ^ rhs.item(i);
    }
    return result;
}

template <typename T, std::size_t Dim, typename Derived>
const NdArray<T, Dim> operator^(const NdArrayBase<T, Dim, Derived> &lhs, const T &rhs) {
    NdArray<T, Dim> result(lhs.shape());
    std::size_t size = lhs.shape().size();
    for (std::size_t i = 0; i < size; ++i) {
        result.item(i) = lhs.item(i) ^ rhs;
    }
    return result;
}

template <typename T, std::size_t Dim, typename Derived1, typename Derived2>
const NdArray<T, Dim> operator|(const NdArrayBase<T, Dim, Derived1> &lhs, const NdArrayBase<T, Dim, Derived2> &rhs) {
    util::validate_shape_binary_op(lhs.shape(), rhs.shape());

    NdArray<T, Dim> result(lhs.shape());
    std::size_t size = lhs.shape().size();
    for (std::size_t i = 0; i < size; ++i) {
        result.item(i) = lhs.item(i) | rhs.item(i);
    }
    return result;
}

template <typename T, std::size_t Dim, typename Derived>
const NdArray<T, Dim> operator|(const T &lhs, const NdArrayBase<T, Dim, Derived> &rhs) {
    NdArray<T, Dim> result(rhs.shape());
    std::size_t size = rhs.shape().size();
    for (std::size_t i = 0; i < size; ++i) {
        result.item(i) = lhs | rhs.item(i);
    }
    return result;
}

template <typename T, std::size_t Dim, typename Derived>
const NdArray<T, Dim> operator|(const NdArrayBase<T, Dim, Derived> &lhs, const T &rhs) {
    NdArray<T, Dim> result(lhs.shape());
    std::size_t size = lhs.shape().size();
    for (std::size_t i = 0; i < size; ++i) {
        result.item(i) = lhs.item(i) | rhs;
    }
    return result;
}

/* Compound assignment operators **************************************************************************************/

template <typename T, std::size_t Dim, typename Derived1, typename Derived2>
NdArrayBase<T, Dim, Derived1> &operator+=(NdArrayBase<T, Dim, Derived1> &lhs,
                                          const NdArrayBase<T, Dim, Derived2> &rhs) {
    util::validate_shape_binary_op(lhs.shape(), rhs.shape());

    std::size_t size = lhs.shape().size();
    for (std::size_t i = 0; i < size; ++i) {
        lhs.item(i) += rhs.item(i);
    }
    return lhs;
}

template <typename T, std::size_t Dim, typename Derived>
NdArrayBase<T, Dim, Derived> &operator+=(NdArrayBase<T, Dim, Derived> &lhs, const T &rhs) {
    std::size_t size = lhs.shape().size();
    for (std::size_t i = 0; i < size; ++i) {
        lhs.item(i) += rhs;
    }
    return lhs;
}

template <typename T, std::size_t Dim, typename Derived1, typename Derived2>
NdArrayBase<T, Dim, Derived1> &operator-=(NdArrayBase<T, Dim, Derived1> &lhs,
                                          const NdArrayBase<T, Dim, Derived2> &rhs) {
    util::validate_shape_binary_op(lhs.shape(), rhs.shape());

    std::size_t size = lhs.shape().size();
    for (std::size_t i = 0; i < size; ++i) {
        lhs.item(i) -= rhs.item(i);
    }
    return lhs;
}

template <typename T, std::size_t Dim, typename Derived>
NdArrayBase<T, Dim, Derived> &operator-=(NdArrayBase<T, Dim, Derived> &lhs, const T &rhs) {
    std::size_t size = lhs.shape().size();
    for (std::size_t i = 0; i < size; ++i) {
        lhs.item(i) -= rhs;
    }
    return lhs;
}

template <typename T, std::size_t Dim, typename Derived1, typename Derived2>
NdArrayBase<T, Dim, Derived1> &operator*=(NdArrayBase<T, Dim, Derived1> &lhs,
                                          const NdArrayBase<T, Dim, Derived2> &rhs) {
    util::validate_shape_binary_op(lhs.shape(), rhs.shape());

    std::size_t size = lhs.shape().size();
    for (std::size_t i = 0; i < size; ++i) {
        lhs.item(i) *= rhs.item(i);
    }
    return lhs;
}

template <typename T, std::size_t Dim, typename Derived>
NdArrayBase<T, Dim, Derived> &operator*=(NdArrayBase<T, Dim, Derived> &lhs, const T &rhs) {
    std::size_t size = lhs.shape().size();
    for (std::size_t i = 0; i < size; ++i) {
        lhs.item(i) *= rhs;
    }
    return lhs;
}

template <typename T, std::size_t Dim, typename Derived1, typename Derived2>
NdArrayBase<T, Dim, Derived1> &operator/=(NdArrayBase<T, Dim, Derived1> &lhs,
                                          const NdArrayBase<T, Dim, Derived2> &rhs) {
    util::validate_shape_binary_op(lhs.shape(), rhs.shape());

    std::size_t size = lhs.shape().size();
    for (std::size_t i = 0; i < size; ++i) {
        lhs.item(i) /= rhs.item(i);
    }
    return lhs;
}

template <typename T, std::size_t Dim, typename Derived>
NdArrayBase<T, Dim, Derived> &operator/=(NdArrayBase<T, Dim, Derived> &lhs, const T &rhs) {
    std::size_t size = lhs.shape().size();
    for (std::size_t i = 0; i < size; ++i) {
        lhs.item(i) /= rhs;
    }
    return lhs;
}

template <typename T, std::size_t Dim, typename Derived1, typename Derived2>
NdArrayBase<T, Dim, Derived1> &operator%=(NdArrayBase<T, Dim, Derived1> &lhs,
                                          const NdArrayBase<T, Dim, Derived2> &rhs) {
    util::validate_shape_binary_op(lhs.shape(), rhs.shape());

    std::size_t size = lhs.shape().size();
    for (std::size_t i = 0; i < size; ++i) {
        if constexpr (std::is_floating_point_v<T>) {
            lhs.item(i) = std::fmod(lhs.item(i), rhs.item(i));
        } else {
            lhs.item(i) %= rhs.item(i);
        }
    }
    return lhs;
}

template <typename T, std::size_t Dim, typename Derived>
NdArrayBase<T, Dim, Derived> &operator%=(NdArrayBase<T, Dim, Derived> &lhs, const T &rhs) {
    std::size_t size = lhs.shape().size();
    for (std::size_t i = 0; i < size; ++i) {
        if constexpr (std::is_floating_point_v<T>) {
            lhs.item(i) = std::fmod(lhs.item(i), rhs);
        } else {
            lhs.item(i) %= rhs;
        }
    }
}

template <typename T, std::size_t Dim, typename Derived1, typename Derived2>
NdArrayBase<T, Dim, Derived1> &operator<<=(NdArrayBase<T, Dim, Derived1> &lhs,
                                           const NdArrayBase<T, Dim, Derived2> &rhs) {
    util::validate_shape_binary_op(lhs.shape(), rhs.shape());

    std::size_t size = lhs.shape().size();
    for (std::size_t i = 0; i < size; ++i) {
        lhs.item(i) <<= rhs.item(i);
    }
    return lhs;
}

template <typename T, std::size_t Dim, typename Derived>
NdArrayBase<T, Dim, Derived> &operator<<=(NdArrayBase<T, Dim, Derived> &lhs, const T &rhs) {
    std::size_t size = lhs.shape().size();
    for (std::size_t i = 0; i < size; ++i) {
        lhs.item(i) <<= rhs;
    }
    return lhs;
}

template <typename T, std::size_t Dim, typename Derived1, typename Derived2>
NdArrayBase<T, Dim, Derived1> &operator>>=(NdArrayBase<T, Dim, Derived1> &lhs,
                                           const NdArrayBase<T, Dim, Derived2> &rhs) {
    util::validate_shape_binary_op(lhs.shape(), rhs.shape());

    std::size_t size = lhs.shape().size();
    for (std::size_t i = 0; i < size; ++i) {
        lhs.item(i) >>= rhs.item(i);
    }
    return lhs;
}

template <typename T, std::size_t Dim, typename Derived>
NdArrayBase<T, Dim, Derived> &operator>>=(NdArrayBase<T, Dim, Derived> &lhs, const T &rhs) {
    std::size_t size = lhs.shape().size();
    for (std::size_t i = 0; i < size; ++i) {
        lhs.item(i) >>= rhs;
    }
    return lhs;
}

template <typename T, std::size_t Dim, typename Derived1, typename Derived2>
NdArrayBase<T, Dim, Derived1> &operator&=(NdArrayBase<T, Dim, Derived1> &lhs,
                                          const NdArrayBase<T, Dim, Derived2> &rhs) {
    util::validate_shape_binary_op(lhs.shape(), rhs.shape());

    std::size_t size = lhs.shape().size();
    for (std::size_t i = 0; i < size; ++i) {
        lhs.item(i) &= rhs.item(i);
    }
    return lhs;
}

template <typename T, std::size_t Dim, typename Derived>
NdArrayBase<T, Dim, Derived> &operator&=(NdArrayBase<T, Dim, Derived> &lhs, const T &rhs) {
    std::size_t size = lhs.shape().size();
    for (std::size_t i = 0; i < size; ++i) {
        lhs.item(i) &= rhs;
    }
    return lhs;
}

template <typename T, std::size_t Dim, typename Derived1, typename Derived2>
NdArrayBase<T, Dim, Derived1> &operator^=(NdArrayBase<T, Dim, Derived1> &lhs,
                                          const NdArrayBase<T, Dim, Derived2> &rhs) {
    util::validate_shape_binary_op(lhs.shape(), rhs.shape());

    std::size_t size = lhs.shape().size();
    for (std::size_t i = 0; i < size; ++i) {
        lhs.item(i) ^= rhs.item(i);
    }
    return lhs;
}

template <typename T, std::size_t Dim, typename Derived>
NdArrayBase<T, Dim, Derived> &operator^=(NdArrayBase<T, Dim, Derived> &lhs, const T &rhs) {
    std::size_t size = lhs.shape().size();
    for (std::size_t i = 0; i < size; ++i) {
        lhs.item(i) ^= rhs;
    }
    return lhs;
}

template <typename T, std::size_t Dim, typename Derived1, typename Derived2>
NdArrayBase<T, Dim, Derived1> &operator|=(NdArrayBase<T, Dim, Derived1> &lhs,
                                          const NdArrayBase<T, Dim, Derived2> &rhs) {
    util::validate_shape_binary_op(lhs.shape(), rhs.shape());

    std::size_t size = lhs.shape().size();
    for (std::size_t i = 0; i < size; ++i) {
        lhs.item(i) |= rhs.item(i);
    }
    return lhs;
}

template <typename T, std::size_t Dim, typename Derived>
NdArrayBase<T, Dim, Derived> &operator|=(NdArrayBase<T, Dim, Derived> &lhs, const T &rhs) {
    std::size_t size = lhs.shape().size();
    for (std::size_t i = 0; i < size; ++i) {
        lhs.item(i) |= rhs;
    }
    return lhs;
}

}  // namespace ndarray

#endif
