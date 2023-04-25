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

template <typename T, std::size_t Dim, typename Derived>
class NdArrayBase {
public:
    using dtype = T;
    static constexpr std::size_t dim = Dim;

    NdArrayBase() = default;
    NdArrayBase(const Shape<Dim> &shape) : shape(shape) {}

    /* Indexing and slicing *******************************************************************************************/

    template <typename... Args>
    auto operator[](Args... args) {
        return static_cast<Derived *>(this)->operator[](args...);
    }

    template <typename... Args>
    auto operator[](Args... args) const {
        return static_cast<const Derived *>(this)->operator[](args...);
    }

    /* Indexing and slicing *******************************************************************************************/

    auto operator=(const NdArray<T, Dim> &other) {
        return static_cast<Derived *>(this)->operator=(other);
    }

    auto operator=(const T &val) {
        return static_cast<Derived *>(this)->operator=(val);
    }

    /* Methods ********************************************************************************************************/

    bool all(void) const {
        return static_cast<const Derived *>(this)->all();
    }

    bool any(void) const {
        return static_cast<const Derived *>(this)->any();
    }

    template <typename U>
    NdArray<U, Dim> astype(void) const {
        return static_cast<const Derived *>(this)->template astype<U>();
    }

    void fill(const T &val) {
        static_cast<Derived *>(this)->fill(val);
    }

    T &item(index_t index) {
        return static_cast<Derived *>(this)->item(index);
    }

    const T &item(index_t index) const {
        return static_cast<const Derived *>(this)->item(index);
    }

    std::size_t itemsize(void) const {
        return sizeof(T);
    }

    std::size_t nbytes(void) const {
        return this->size() * this->itemsize();
    }

    index_t size(void) const {
        return this->shape.size();
    }

    const Shape<Dim> shape;
};

}  // namespace ndarray

#endif
