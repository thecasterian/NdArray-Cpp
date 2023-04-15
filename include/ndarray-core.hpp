#ifndef NDARRAY_CORE_HPP
#define NDARRAY_CORE_HPP

#include "ndarray-expr.hpp"
#include "ndarray-size.hpp"
#include "ndarray-slice.hpp"

namespace ndarray {

template <typename T, std::size_t Dim>
class NdArray : public NdArrayExpr<T, Dim, NdArray<T, Dim>> {
public:
    using NdArrayExpr<T, Dim, NdArray<T, Dim>>::operator[];

    NdArray(const Size<Dim> &shape) : NdArrayExpr<T, Dim, NdArray<T, Dim>>(shape), data(new T[shape.numel()]) {}

    NdArray(const std::initializer_list<NdArray<T, Dim - 1>> &list)
        requires(Dim > 1)
        : NdArrayExpr<T, Dim, NdArray<T, Dim>>(Size<Dim>(static_cast<index_t>(list.size()), list.begin()->shape)),
          data(new T[this->shape.numel()]) {
        const Size<Dim - 1> &sub_shape = list.begin()->shape;
        for (const NdArray<T, Dim - 1> &sub_array : list) {
            if (sub_array.shape != sub_shape)
                throw std::invalid_argument("Invalid shape of initializer list");
        }

        auto data_ptr = data;
        for (auto &sub_array : list) {
            std::copy(sub_array.data, sub_array.data + sub_array.shape.numel(), data_ptr);
            data_ptr += sub_array.shape.numel();
        }
    }

    NdArray(const std::initializer_list<T> &list)
        requires(Dim == 1)
        : NdArrayExpr<T, Dim, NdArray<T, Dim>>(Size<1>({static_cast<index_t>(list.size())})), data(new T[list.size()]) {
        if (list.size() == 0) {
            throw std::invalid_argument("Length of initializer list cannot be 0");
        }

        std::copy(list.begin(), list.end(), data);
    }

    ~NdArray() {
        delete[] data;
    }

    NdArray(const NdArray &other)
        : NdArrayExpr<T, Dim, NdArray<T, Dim>>(other.shape), data(new T[other.shape.numel()]) {
        std::copy(other.data, other.data + other.shape.numel(), this->data);
    }

    NdArray(NdArray &&other) : NdArrayExpr<T, Dim, NdArray<T, Dim>>(other.shape), data(other.data) {
        other.data = nullptr;
    }

    NdArray &operator=(const NdArray &other) {
        if (this != &other) {
            delete[] this->data;
            this->shape = other.shape;
            this->data = new T[other.shape.numel()];
            std::copy(other.data, other.data + other.shape.numel(), this->data);
        }

        return *this;
    }

    NdArray &operator=(NdArray &&other) {
        if (this != &other) {
            delete[] this->data;
            this->shape = other.shape;
            this->data = other.data;
            other.data = nullptr;
        }

        return *this;
    }

    template <typename... Args>
        requires(sizeof...(Args) == Dim && (util::is_index_type<Args> && ...))
    T &operator[](Args... args) {
        std::array<index_t, Dim> arg_array = {args...};
        return this->operator[](arg_array);
    }

    template <typename... Args>
        requires(sizeof...(Args) == Dim && (util::is_index_type<Args> && ...))
    T operator[](Args... args) const {
        std::array<index_t, Dim> arg_array = {args...};
        return this->operator[](arg_array);
    }

    T &operator[](const std::array<index_t, Dim> &indices) {
        this->validate_indices(indices);

        index_t index = 0;
        for (std::size_t i = 0; i < Dim; ++i) {
            index += (indices[i] >= 0 ? indices[i] : indices[i] + static_cast<index_t>(Dim)) * this->shape.partial[i];
        }

        return data[index];
    }

    T operator[](const std::array<index_t, Dim> &indices) const {
        this->validate_indices(indices);

        index_t index = 0;
        for (std::size_t i = 0; i < Dim; ++i) {
            index += (indices[i] >= 0 ? indices[i] : indices[i] + static_cast<index_t>(Dim)) * this->shape.partial[i];
        }

        return data[index];
    }

private:
    template <typename, std::size_t>
    friend class NdArray;

    T *data;
};

template <typename T>
class NdArray<T, 0>;

}  // namespace ndarray

#endif
