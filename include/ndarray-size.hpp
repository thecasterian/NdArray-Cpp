#ifndef NDARRAY_SIZE_HPP
#define NDARRAY_SIZE_HPP

#include <iostream>
#include <numeric>
#include <stdexcept>

#include "ndarray-definition.hpp"
#include "ndarray-util.hpp"

namespace ndarray {

template <typename T, std::size_t Dim>
class NdArray;

template <typename T, std::size_t Dim, typename BaseArray>
class NdArraySlice;

template <std::size_t Dim>
class Size {
public:
    Size(const Size &) = default;
    Size(Size &&) = default;
    Size &operator=(const Size &) = default;
    Size &operator=(Size &&) = default;
    ~Size() = default;

    Size(void) {
        std::fill(this->size, this->size + Dim, 1);
        this->init_partial();
    }

    Size(const index_t *size) {
        std::copy(size, size + Dim, this->size);
        this->init_partial();
    }

    Size(const std::initializer_list<index_t> &size) {
        if (size.size() != Dim)
            throw std::invalid_argument("Invalid length of initializer list");
        std::copy(size.begin(), size.end(), this->size);
        this->init_partial();
    }

    index_t operator[](std::size_t i) const {
        if (i >= Dim)
            throw std::out_of_range("Index out of range");
        return this->size[i];
    }

    bool operator==(const Size<Dim> &other) const {
        return std::equal(this->size, this->size + Dim, other.size);
    }

    bool operator!=(const Size &other) const {
        return !(*this == other);
    }

    operator std::string() const {
        std::string str = "(";
        for (std::size_t i = 0; i < Dim; ++i) {
            str += std::to_string(this->size[i]);
            if (i != Dim - 1)
                str += ", ";
        }
        str += ")";
        return str;
    }

private:
    template <typename, std::size_t>
    friend class NdArray;

    template <typename, std::size_t, typename>
    friend class NdArraySlice;

    template <std::size_t>
    friend class Size;

    Size(index_t size_first, const Size<Dim - 1> &size_rest) {
        this->size[0] = size_first;
        std::copy(size_rest.size, size_rest.size + Dim - 1, this->size + 1);
        this->init_partial();
    }

    void init_partial(void) {
        this->partial[Dim - 1] = 1;
        for (std::size_t i = Dim - 1; i > 0; --i) {
            this->partial[i - 1] = this->partial[i] * this->size[i];
        }
    }

    std::size_t numel(void) const {
        return std::accumulate(this->size, this->size + Dim, 1, std::multiplies<index_t>());
    }

    index_t size[Dim];
    index_t partial[Dim];
};

template <std::size_t Dim>
std::ostream &operator<<(std::ostream &os, const Size<Dim> &size) {
    os << static_cast<std::string>(size);
    return os;
}

}  // namespace ndarray

#endif