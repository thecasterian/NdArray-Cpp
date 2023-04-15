#ifndef NDARRAY_EXPR_HPP
#define NDARRAY_EXPR_HPP

#include <type_traits>

#include "ndarray-size.hpp"
#include "ndarray-slice.hpp"

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
        return static_cast<const Derived &>(*this).operator()(args...);
    }

    template <typename... Args>
        requires(sizeof...(Args) == Dim && !(util::is_index_type<Args> && ...) &&
                 (util::is_index_slice_type<Args> && ...))
    NdArraySlice<T, util::count_slice_type<Args...>, NdArrayExpr<T, Dim, Derived>> operator[](Args... args) {
        std::array<index_t, util::count_slice_type<Args...>> base_axes;
        std::array<Slice, util::count_slice_type<Args...>> slices;

        index_t i = -1, j = -1;
        ((++i, util::is_slice_type<Args> ? (++j, (slices[j] = args), (base_axes[j] = i)) : 0), ...);

        return {static_cast<NdArrayExpr<T, Dim, Derived> &>(*this), base_axes, slices};
    }

    template <typename... Args>
        requires(sizeof...(Args) < Dim)
    NdArraySlice<T, util::count_slice_type<Args...> + Dim - sizeof...(Args), NdArrayExpr<T, Dim, Derived>> operator[](
        Args... args) {
        std::array<index_t, util::count_slice_type<Args...> + Dim - sizeof...(Args)> base_axes;
        std::array<Slice, util::count_slice_type<Args...> + Dim - sizeof...(Args)> slices;

        index_t i = -1, j = -1;
        ((++i, util::is_slice_type<Args> ? (++j, (slices[j] = args), (base_axes[j] = i)) : 0), ...);

        for (std::size_t k = 0; k < Dim - sizeof...(Args); ++k) {
            ++i;
            ++j;
            base_axes[j] = i;
            slices[j] = Slice();
        }
    }

    Size<Dim> size(void) const {
        return this->shape;
    }

    Size<Dim> shape;
};

}  // namespace ndarray

#endif
