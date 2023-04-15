#ifndef NDARRAY_EXPR_HPP
#define NDARRAY_EXPR_HPP

#include <iostream>
#include <type_traits>

#include "ndarray-size.hpp"
#include "ndarray-slice.hpp"
#include "ndarray-util.hpp"

namespace ndarray {

template <typename T, std::size_t Dim>
class NdArray;

template <typename T, std::size_t Dim, typename Derived>
class NdArrayExpr {
public:
    using dtype = T;
    static constexpr std::size_t dim = Dim;

    NdArrayExpr() = default;
    NdArrayExpr(Size<Dim> shape) : shape(shape) {}

    template <typename... Args>
        requires(sizeof...(Args) == Dim && (util::is_index_type<Args> && ...))
    T operator[](Args... args) const {
        return static_cast<const Derived &>(*this).operator[](args...);
    }

    T operator[](const std::array<index_t, Dim> &indices) const {
        return static_cast<const Derived &>(*this).operator[](indices);
    }

    template <typename... Args>
        requires(sizeof...(Args) <= Dim && (util::is_index_slice_type<Args> && ...))
    NdArraySlice<T, util::count_slice_type<Args...> + Dim - sizeof...(Args), Derived> operator[](Args... args) {
        std::array<bool, Dim> is_slice_axis;
        std::array<index_t, sizeof...(Args) - util::count_slice_type<Args...>> indices;
        std::array<Slice, util::count_slice_type<Args...> + Dim - sizeof...(Args)> slices;

        index_t i = 0;
        ((is_slice_axis[i++] = util::is_slice_type<Args>), ...);
        for (std::size_t i = sizeof...(Args); i < Dim; ++i) {
            is_slice_axis[i] = true;
        }

        util::separate_index_slice<sizeof...(Args) - util::count_slice_type<Args...>,
                                   util::count_slice_type<Args...> + Dim - sizeof...(Args), Args...>(
            indices.begin(), slices.begin(), args...);

        return {static_cast<Derived &>(*this), is_slice_axis, indices, slices};
    }

    Size<Dim> size(void) const {
        return this->shape;
    }

    Size<Dim> shape;

protected:
    void validate_indices(const std::array<index_t, Dim> &indices) const {
        for (std::size_t i = 0; i < Dim; ++i) {
            if (indices[i] < -static_cast<index_t>(Dim) || indices[i] >= this->shape.size[i]) {
                throw std::out_of_range("Index " + std::to_string(indices[i]) + " is out of range for axis " +
                                        std::to_string(i) + " with size " + std::to_string(this->shape.size[i]));
            }
        }
    }
};

}  // namespace ndarray

#endif
