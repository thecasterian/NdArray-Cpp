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
    NdArrayBase(const Shape<Dim> &shape) : _shape(shape) {}

    operator std::string() const {
        return this->to_string();
    }

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
    NdArray<U, Dim> as_type(void) const {
        return static_cast<const Derived *>(this)->template as_type<U>();
    }

    void fill(const T &val) {
        static_cast<Derived *>(this)->fill(val);
    }

    NdArray<T, 1> flatten(void) const {
        return static_cast<const Derived *>(this)->flatten();
    }

    T &item(index_t index) {
        return static_cast<Derived *>(this)->item(index);
    }

    const T &item(index_t index) const {
        return static_cast<const Derived *>(this)->item(index);
    }

    std::size_t item_size(void) const {
        return sizeof(T);
    }

    std::size_t nbytes(void) const {
        return this->_shape.size() * sizeof(T);
    }

    template <std::size_t NewDim>
    NdArray<T, NewDim> reshape(const Shape<NewDim> &new_shape) const {
        return static_cast<const Derived *>(this)->reshape(new_shape);
    }

    NdArray<T, 1> ravel(void) const {
        return static_cast<const Derived *>(this)->flatten();
    }

    const Shape<Dim> &shape(void) const {
        return this->_shape;
    }

    index_t size(void) const {
        return this->_shape.size();
    }

    std::string to_string(void) const {
        std::string result = "NdArray(";
        result += this->to_string_helper();
        result += ")";

        return result;
    }

    util::nested_vector_t<T, Dim> to_vector(void) const {
        util::nested_vector_t<T, Dim> result;
        result.reserve(this->_shape[0]);
        for (index_t i = 0; i < this->_shape[0]; ++i) {
            if constexpr (Dim == 1) {
                result.push_back(this->item(i));
            } else {
                result.push_back(this->operator[](i).to_vector());
            }
        }

        return result;
    }

private:
    template <typename, std::size_t, typename>
    friend class NdArrayBase;

    template <typename, std::size_t>
    friend class NdArray;

    template <typename, std::size_t, typename>
    friend class NdArraySlice;

    std::string to_string_helper(void) const {
        std::string result = "{";
        for (index_t i = 0; i < this->_shape[0]; ++i) {
            if constexpr (Dim == 1) {
                if constexpr (std::is_arithmetic_v<T>) {
                    result += std::to_string(this->item(i));
                } else {
                    result += static_cast<std::string>(this->item(i));
                }
            } else {
                result += this->operator[](i).to_string_helper();
            }

            if (i != this->_shape[0] - 1) {
                result += ", ";
            }
        }
        result += "}";

        return result;
    }

    Shape<Dim> _shape;
};

template <typename T, std::size_t Dim, typename Derived>
std::ostream &operator<<(std::ostream &os, const NdArrayBase<T, Dim, Derived> &ndarray_base) {
    os << ndarray_base.to_string();
    return os;
}

}  // namespace ndarray

#endif
