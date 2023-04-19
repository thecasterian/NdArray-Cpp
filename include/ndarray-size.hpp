#ifndef NDARRAY_SIZE_HPP
#define NDARRAY_SIZE_HPP

#include <array>
#include <iostream>
#include <numeric>
#include <stdexcept>

#include "ndarray-definition.hpp"
#include "ndarray-util.hpp"

namespace ndarray {

template <typename T, std::size_t Dim>
class NdArrayBase;

template <typename T, std::size_t Dim>
class NdArray;

template <typename T, std::size_t Dim, typename Operand>
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
        std::fill(this->size.begin(), this->size.end(), 1);
        this->init_partial();
    }

    Size(const index_t *size) {
        std::copy(size, size + Dim, this->size.begin());
        this->init_partial();
    }

    Size(const std::initializer_list<index_t> &size) {
        if (size.size() != Dim)
            throw std::invalid_argument("Invalid length of initializer list");
        std::copy(size.begin(), size.end(), this->size.begin());
        this->init_partial();
    }

    Size(index_t size_first, const Size<Dim - 1> &size_rest) {
        this->size[0] = size_first;
        std::copy(size_rest.size.begin(), size_rest.size.end(), this->size.begin() + 1);
        this->init_partial();
    }

    index_t operator[](std::size_t i) const {
        if (i >= Dim)
            throw std::out_of_range("Index out of range");
        return this->size[i];
    }

    bool operator==(const Size<Dim> &other) const {
        return std::equal(this->size.begin(), this->size.end(), other.size.begin());
    }

    bool operator!=(const Size &other) const {
        return !(*this == other);
    }

    std::string to_string() const {
        std::string str = "Size(";
        for (std::size_t i = 0; i < Dim; ++i) {
            str += std::to_string(this->size[i]);
            if (i != Dim - 1)
                str += ", ";
        }
        str += ")";
        return str;
    }

    operator std::string() const {
        return this->to_string();
    }

    std::size_t numel(void) const {
        return std::accumulate(this->size.begin(), this->size.end(), 1, std::multiplies<index_t>());
    }

private:
    template <typename, std::size_t>
    friend class NdArrayBase;

    template <typename, std::size_t>
    friend class NdArray;

    template <typename, std::size_t, typename>
    friend class NdArraySlice;

    template <std::size_t>
    friend class Size;

    void init_partial(void) {
        this->partial[Dim - 1] = 1;
        for (std::size_t i = Dim - 1; i > 0; --i) {
            this->partial[i - 1] = this->partial[i] * this->size[i];
        }
    }

    std::array<index_t, Dim> size;
    std::array<index_t, Dim> partial;
};

template <std::size_t Dim>
std::ostream &operator<<(std::ostream &os, const Size<Dim> &size) {
    os << size.to_string();
    return os;
}

}  // namespace ndarray

#endif
