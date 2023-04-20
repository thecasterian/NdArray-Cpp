#ifndef NDARRAY_EXPR_HPP
#define NDARRAY_EXPR_HPP

#include <iostream>
#include <type_traits>

#include "ndarray-shape.hpp"
#include "ndarray-slice.hpp"
#include "ndarray-util.hpp"

namespace ndarray {

template <typename T, std::size_t Dim>
class NdArray;

template <typename T, std::size_t Dim>
class NdArrayBase {
public:
    using dtype = T;
    static constexpr std::size_t dim = Dim;

    NdArrayBase() = default;
    NdArrayBase(Shape<Dim> shape) : shape(shape) {}

    index_t size(void) const {
        return this->shape.size();
    }

    const Shape<Dim> shape;
};

}  // namespace ndarray

#endif
