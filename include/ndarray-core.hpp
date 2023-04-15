#ifndef NDARRAY_CORE_HPP
#define NDARRAY_CORE_HPP

#include "ndarray-expr.hpp"
#include "ndarray-size.hpp"
#include "ndarray-slice.hpp"

namespace ndarray {

template <typename T, std::size_t Dim>
class NdArray : public NdArrayExpr<T, Dim, NdArray<T, Dim>> {
public:
    NdArray(const Size<Dim> &shape) : NdArrayExpr<T, Dim, NdArray<T, Dim>>(shape), data(new T[shape.numel()]) {}

    NdArray(const std::initializer_list<NdArray<T, Dim - 1>> &list)
        requires(Dim > 1)
        : NdArrayExpr<T, Dim, NdArray<T, Dim>>(Size(static_cast<index_t>(list.size()), list.begin()->shape)),
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
        index_t arg;
        int i = -1;
        bool is_not_out_of_range =
            ((++i, (arg = args), -static_cast<index_t>(Dim) <= args && args < this->shape.size[i]) && ...);
        if (!is_not_out_of_range) {
            throw std::out_of_range("Index " + std::to_string(arg) + " is out of range for axis " + std::to_string(i) +
                                    " with size " + std::to_string(this->shape.size[i]));
        }

        index_t idx = 0;
        i = -1;
        ((++i, idx += (args >= 0 ? args : args + this->shape.size[i]) * this->shape.partial[i]), ...);
        return data[idx];
    }

    template <typename... Args>
        requires(sizeof...(Args) == Dim && (util::is_index_type<Args> && ...))
    T operator[](Args... args) const {
        index_t arg;
        int i = -1;
        bool is_not_out_of_range =
            ((++i, (arg = args), -static_cast<index_t>(Dim) <= args && args < this->shape.size[i]) && ...);
        if (!is_not_out_of_range) {
            throw std::out_of_range("Index " + std::to_string(arg) + " is out of range for axis " + std::to_string(i) +
                                    " with size " + std::to_string(this->shape.size[i]));
        }

        index_t idx = 0;
        i = -1;
        ((++i, idx += (args >= 0 ? args : args + this->shape.size[i]) * this->shape.partial[i]), ...);
        return data[idx];
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
