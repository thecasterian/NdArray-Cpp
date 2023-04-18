#ifndef NDARRAY_CORE_HPP
#define NDARRAY_CORE_HPP

#include "ndarray-base.hpp"
#include "ndarray-size.hpp"
#include "ndarray-slice.hpp"

namespace ndarray {

template <typename T, std::size_t Dim>
class NdArray : public NdArrayBase<T, Dim> {
public:
    NdArray(const Size<Dim> &shape) : NdArrayBase<T, Dim>(shape), data(new T[shape.numel()]) {}

    NdArray(const std::initializer_list<NdArray<T, Dim - 1>> &list)
        requires(Dim > 1)
        : NdArrayBase<T, Dim>(Size<Dim>(static_cast<index_t>(list.size()), list.begin()->shape)),
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
        : NdArrayBase<T, Dim>(Size<1>({static_cast<index_t>(list.size())})), data(new T[list.size()]) {
        if (list.size() == 0) {
            throw std::invalid_argument("Length of initializer list cannot be 0");
        }

        std::copy(list.begin(), list.end(), data);
    }

    ~NdArray() {
        delete[] data;
    }

    NdArray(const NdArray &other) : NdArrayBase<T, Dim>(other.shape), data(new T[other.shape.numel()]) {
        std::copy(other.data, other.data + other.shape.numel(), this->data);
    }

    NdArray(NdArray &&other) : NdArrayBase<T, Dim>(other.shape), data(other.data) {
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

    /* Indexing *******************************************************************************************************/

    template <typename... Args>
        requires(sizeof...(Args) == Dim && (util::is_index_type<Args> && ...))
    T &operator[](Args... args) {
        std::array<index_t, Dim> arg_array = {args...};
        return this->operator[](arg_array);
    }

    template <typename... Args>
        requires(sizeof...(Args) == Dim && (util::is_index_type<Args> && ...))
    const T &operator[](Args... args) const {
        std::array<index_t, Dim> arg_array = {args...};
        return this->operator[](arg_array);
    }

    T &operator[](const std::array<index_t, Dim> &indices) {
        std::array<index_t, Dim> normalized_indices = util::normalize_indices(this->shape, indices);

        index_t index = 0;
        for (std::size_t i = 0; i < Dim; ++i) {
            index += normalized_indices[i] * this->shape.partial[i];
        }

        return data[index];
    }

    const T &operator[](const std::array<index_t, Dim> &indices) const {
        std::array<index_t, Dim> normalized_indices = util::normalize_indices(this->shape, indices);

        index_t index = 0;
        for (std::size_t i = 0; i < Dim; ++i) {
            index += normalized_indices[i] * this->shape.partial[i];
        }

        return data[index];
    }

    /* Slicing ********************************************************************************************************/

    template <typename... Args>
        requires(sizeof...(Args) <= Dim && (util::is_index_slice_type<Args> && ...) &&
                 !(sizeof...(Args) == Dim && (util::is_index_type<Args> && ...)))
    NdArraySlice<T, util::count_slice_type<Args...> + Dim - sizeof...(Args), NdArray<T, Dim>> operator[](Args... args) {
        static constexpr std::size_t NIndices = sizeof...(Args) - util::count_slice_type<Args...>;
        static constexpr std::size_t NSlices = util::count_slice_type<Args...> + Dim - sizeof...(Args);

        std::array<bool, Dim> is_slice_axis;
        std::array<index_t, NIndices> indices;
        std::array<Slice, NSlices> slices;

        index_t i = 0;
        ((is_slice_axis[i++] = util::is_slice_type<Args>), ...);
        for (std::size_t i = sizeof...(Args); i < Dim; ++i) {
            is_slice_axis[i] = true;
        }

        util::separate_index_slice<NIndices, NSlices, Args...>(indices.begin(), slices.begin(), args...);

        util::normalize_indices_slices<NIndices, NSlices>(this->shape, is_slice_axis, indices, slices);

        return {static_cast<NdArray<T, Dim> &>(*this), is_slice_axis, indices, slices};
    }

    template <typename... Args>
        requires(sizeof...(Args) <= Dim && (util::is_index_slice_type<Args> && ...) &&
                 !(sizeof...(Args) == Dim && (util::is_index_type<Args> && ...)))
    const NdArraySlice<T, util::count_slice_type<Args...> + Dim - sizeof...(Args), const NdArray<T, Dim>> operator[](
        Args... args) const {
        static constexpr std::size_t NIndices = sizeof...(Args) - util::count_slice_type<Args...>;
        static constexpr std::size_t NSlices = util::count_slice_type<Args...> + Dim - sizeof...(Args);

        std::array<bool, Dim> is_slice_axis;
        std::array<index_t, NIndices> indices;
        std::array<Slice, NSlices> slices;

        index_t i = 0;
        ((is_slice_axis[i++] = util::is_slice_type<Args>), ...);
        for (std::size_t i = sizeof...(Args); i < Dim; ++i) {
            is_slice_axis[i] = true;
        }

        util::separate_index_slice<NIndices, NSlices, Args...>(indices.begin(), slices.begin(), args...);

        util::normalize_indices_slices<NIndices, NSlices>(this->shape, is_slice_axis, indices, slices);

        return {static_cast<const NdArray<T, Dim> &>(*this), is_slice_axis, indices, slices};
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
