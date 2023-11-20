#ifndef NDARRAY_SIZE_HPP
#define NDARRAY_SIZE_HPP

#include <array>
#include <iostream>
#include <numeric>
#include <stdexcept>

#include "ndarray-definition.hpp"
#include "ndarray-util.hpp"

namespace ndarray {

template <typename T, std::size_t Dim, typename Derived>
class NdArrayBase;

template <typename T, std::size_t Dim>
class NdArray;

template <typename T, std::size_t Dim, typename Operand>
class NdArraySlice;

template <std::size_t Dim>
class Shape {
public:
    Shape(const Shape &) = default;
    Shape(Shape &&) = default;
    Shape &operator=(const Shape &) = default;
    Shape &operator=(Shape &&) = default;
    ~Shape() = default;

    Shape(void) {
        std::fill(this->_shape.begin(), this->_shape.end(), 1);
        this->init_partial();
    }

    Shape(const index_t *shape) {
        std::copy(shape, shape + Dim, this->_shape.begin());
        this->init_partial();
    }

    Shape(const std::array<index_t, Dim> &shape) {
        std::copy(shape.begin(), shape.end(), this->_shape.begin());
        this->init_partial();
    }

    Shape(const std::initializer_list<index_t> &shape) {
        if (shape.size() != Dim)
            throw std::invalid_argument("Invalid length of initializer list");
        std::copy(shape.begin(), shape.end(), this->_shape.begin());
        this->init_partial();
    }

    Shape(index_t shape_first, const Shape<Dim - 1> &shape_rest) {
        this->_shape[0] = shape_first;
        std::copy(shape_rest._shape.begin(), shape_rest._shape.end(), this->_shape.begin() + 1);
        this->init_partial();
    }

    index_t operator[](index_t i) const {
        if (i < -static_cast<index_t>(Dim) || i >= static_cast<index_t>(Dim))
            throw std::out_of_range("Shape index out of range");
        if (i < 0)
            i += Dim;
        return this->_shape[i];
    }

    bool operator==(const Shape<Dim> &other) const {
        return std::equal(this->_shape.begin(), this->_shape.end(), other._shape.begin());
    }

    bool operator!=(const Shape<Dim> &other) const {
        return !(*this == other);
    }

    std::string to_string() const {
        std::string str = "Shape(";
        for (std::size_t i = 0; i < Dim; ++i) {
            str += std::to_string(this->_shape[i]);
            if (i != Dim - 1)
                str += ", ";
        }
        str += ")";
        return str;
    }

    operator std::string() const {
        return this->to_string();
    }

    index_t size(void) const {
        return std::accumulate(this->_shape.begin(), this->_shape.end(), 1, std::multiplies<index_t>());
    }

private:
    template <typename, std::size_t>
    friend class NdArray;

    template <std::size_t>
    friend class Shape;

    void init_partial(void) {
        this->partial[Dim - 1] = 1;
        for (std::size_t i = Dim - 1; i > 0; --i) {
            this->partial[i - 1] = this->partial[i] * this->_shape[i];
        }
    }

    std::array<index_t, Dim> _shape;
    std::array<index_t, Dim> partial;
};

template <std::size_t Dim>
std::ostream &operator<<(std::ostream &os, const Shape<Dim> &size) {
    os << size.to_string();
    return os;
}

}  // namespace ndarray

#endif
