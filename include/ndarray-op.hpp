#ifndef NDARRAY_OP_HPP
#define NDARRAY_OP_HPP

#include "ndarray-base.hpp"
#include "ndarray-core.hpp"

namespace ndarray {

/* Comparison operators ***********************************************************************************************/

template <typename T1, typename T2, std::size_t Dim, typename Derived1, typename Derived2>
const NdArray<util::eq_t<T1, T2>, Dim> operator==(const NdArrayBase<T1, Dim, Derived1> &lhs,
                                                     const NdArrayBase<T2, Dim, Derived2> &rhs) {
    util::validate_shape_binary_op(lhs.shape, rhs.shape);

    NdArray<util::eq_t<T1, T2>, Dim> result(lhs.shape);
    std::size_t size = lhs.shape.size();
    for (std::size_t i = 0; i < size; ++i) {
        result.item(i) = lhs.item(i) == rhs.item(i);
    }
    return result;
}

template <typename T1, typename T2, std::size_t Dim, typename Derived1, typename Derived2>
const NdArray<util::neq_t<T1, T2>, Dim> operator!=(const NdArrayBase<T1, Dim, Derived1> &lhs,
                                                      const NdArrayBase<T2, Dim, Derived2> &rhs) {
    util::validate_shape_binary_op(lhs.shape, rhs.shape);

    NdArray<util::neq_t<T1, T2>, Dim> result(lhs.shape);
    std::size_t size = lhs.shape.size();
    for (std::size_t i = 0; i < size; ++i) {
        result.item(i) = lhs.item(i) != rhs.item(i);
    }
    return result;
}

template <typename T1, typename T2, std::size_t Dim, typename Derived1, typename Derived2>
const NdArray<util::lt_t<T1, T2>, Dim> operator<(const NdArrayBase<T1, Dim, Derived1> &lhs,
                                                    const NdArrayBase<T2, Dim, Derived2> &rhs) {
    util::validate_shape_binary_op(lhs.shape, rhs.shape);

    NdArray<util::lt_t<T1, T2>, Dim> result(lhs.shape);
    std::size_t size = lhs.shape.size();
    for (std::size_t i = 0; i < size; ++i) {
        result.item(i) = lhs.item(i) < rhs.item(i);
    }
    return result;
}

template <typename T1, typename T2, std::size_t Dim, typename Derived1, typename Derived2>
const NdArray<util::gt_t<T1, T2>, Dim> operator>(const NdArrayBase<T1, Dim, Derived1> &lhs,
                                                    const NdArrayBase<T2, Dim, Derived2> &rhs) {
    util::validate_shape_binary_op(lhs.shape, rhs.shape);

    NdArray<util::gt_t<T1, T2>, Dim> result(lhs.shape);
    std::size_t size = lhs.shape.size();
    for (std::size_t i = 0; i < size; ++i) {
        result.item(i) = lhs.item(i) > rhs.item(i);
    }
    return result;
}

template <typename T1, typename T2, std::size_t Dim, typename Derived1, typename Derived2>
const NdArray<util::le_t<T1, T2>, Dim> operator<=(const NdArrayBase<T1, Dim, Derived1> &lhs,
                                                     const NdArrayBase<T2, Dim, Derived2> &rhs) {
    util::validate_shape_binary_op(lhs.shape, rhs.shape);

    NdArray<util::le_t<T1, T2>, Dim> result(lhs.shape);
    std::size_t size = lhs.shape.size();
    for (std::size_t i = 0; i < size; ++i) {
        result.item(i) = lhs.item(i) <= rhs.item(i);
    }
    return result;
}

template <typename T1, typename T2, std::size_t Dim, typename Derived1, typename Derived2>
const NdArray<util::ge_t<T1, T2>, Dim> operator>=(const NdArrayBase<T1, Dim, Derived1> &lhs,
                                                     const NdArrayBase<T2, Dim, Derived2> &rhs) {
    util::validate_shape_binary_op(lhs.shape, rhs.shape);

    NdArray<util::ge_t<T1, T2>, Dim> result(lhs.shape);
    std::size_t size = lhs.shape.size();
    for (std::size_t i = 0; i < size; ++i) {
        result.item(i) = lhs.item(i) >= rhs.item(i);
    }
    return result;
}

/* Binary arithmetic operators ****************************************************************************************/

template <typename LhsT, typename RhsT, std::size_t Dim, typename Derived1, typename Derived2>
const NdArray<util::add_t<LhsT, RhsT>, Dim> operator+(const NdArrayBase<LhsT, Dim, Derived1> &lhs,
                                                         const NdArrayBase<RhsT, Dim, Derived2> &rhs) {
    util::validate_shape_binary_op(lhs.shape, rhs.shape);

    NdArray<util::add_t<LhsT, RhsT>, Dim> result(lhs.shape);
    std::size_t size = lhs.shape.size();
    for (std::size_t i = 0; i < size; ++i) {
        result.item(i) = lhs.item(i) + rhs.item(i);
    }
    return result;
}

template <typename LhsT, typename RhsT, std::size_t Dim, typename Derived1, typename Derived2>
const NdArray<util::sub_t<LhsT, RhsT>, Dim> operator-(const NdArrayBase<LhsT, Dim, Derived1> &lhs,
                                                         const NdArrayBase<RhsT, Dim, Derived2> &rhs) {
    util::validate_shape_binary_op(lhs.shape, rhs.shape);

    NdArray<util::sub_t<LhsT, RhsT>, Dim> result(lhs.shape);
    std::size_t size = lhs.shape.size();
    for (std::size_t i = 0; i < size; ++i) {
        result.item(i) = lhs.item(i) - rhs.item(i);
    }
    return result;
}

template <typename LhsT, typename RhsT, std::size_t Dim, typename Derived1, typename Derived2>
const NdArray<util::mul_t<LhsT, RhsT>, Dim> operator*(const NdArrayBase<LhsT, Dim, Derived1> &lhs,
                                                         const NdArrayBase<RhsT, Dim, Derived2> &rhs) {
    util::validate_shape_binary_op(lhs.shape, rhs.shape);

    NdArray<util::mul_t<LhsT, RhsT>, Dim> result(lhs.shape);
    std::size_t size = lhs.shape.size();
    for (std::size_t i = 0; i < size; ++i) {
        result.item(i) = lhs.item(i) * rhs.item(i);
    }
    return result;
}

template <typename LhsT, typename RhsT, std::size_t Dim, typename Derived1, typename Derived2>
const NdArray<util::div_t<LhsT, RhsT>, Dim> operator/(const NdArrayBase<LhsT, Dim, Derived1> &lhs,
                                                         const NdArrayBase<RhsT, Dim, Derived2> &rhs) {
    util::validate_shape_binary_op(lhs.shape, rhs.shape);

    NdArray<util::div_t<LhsT, RhsT>, Dim> result(lhs.shape);
    std::size_t size = lhs.shape.size();
    for (std::size_t i = 0; i < size; ++i) {
        result.item(i) = lhs.item(i) / rhs.item(i);
    }
    return result;
}

template <typename LhsT, typename RhsT, std::size_t Dim, typename Derived1, typename Derived2>
const NdArray<util::mod_t<LhsT, RhsT>, Dim> operator%(const NdArrayBase<LhsT, Dim, Derived1> &lhs,
                                                         const NdArrayBase<RhsT, Dim, Derived2> &rhs) {
    util::validate_shape_binary_op(lhs.shape, rhs.shape);

    NdArray<util::mod_t<LhsT, RhsT>, Dim> result(lhs.shape);
    std::size_t size = lhs.shape.size();
    for (std::size_t i = 0; i < size; ++i) {
        result.item(i) = lhs.item(i) % rhs.item(i);
    }
    return result;
}

}  // namespace ndarray

#endif
