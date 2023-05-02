#ifndef NDARRAY_CORE_HPP
#define NDARRAY_CORE_HPP

#include "ndarray-base.hpp"
#include "ndarray-shape.hpp"
#include "ndarray-slice.hpp"

namespace ndarray {

template <typename T, std::size_t Dim>
class NdArray : public NdArrayBase<T, Dim, NdArray<T, Dim>> {
public:
    NdArray(const Shape<Dim> &shape) : NdArrayBase<T, Dim, NdArray<T, Dim>>(shape), _data(new T[shape.size()]) {}

    NdArray(const std::initializer_list<NdArray<T, Dim - 1>> &list)
        requires(Dim > 1)
        : NdArrayBase<T, Dim, NdArray<T, Dim>>(Shape<Dim>(static_cast<index_t>(list.size()), list.begin()->shape)),
          _data(new T[this->shape.size()]) {
        const Shape<Dim - 1> &sub_shape = list.begin()->shape;
        for (const NdArray<T, Dim - 1> &sub_array : list) {
            if (sub_array.shape != sub_shape)
                throw std::invalid_argument("Invalid shape of initializer list");
        }

        auto data_ptr = _data;
        for (auto &sub_array : list) {
            std::copy(sub_array._data, sub_array._data + sub_array.shape.size(), data_ptr);
            data_ptr += sub_array.shape.size();
        }
    }

    NdArray(const std::initializer_list<T> &list)
        requires(Dim == 1)
        : NdArrayBase<T, Dim, NdArray<T, Dim>>(Shape<1>({static_cast<index_t>(list.size())})),
          _data(new T[list.size()]) {
        if (list.size() == 0) {
            throw std::invalid_argument("Length of initializer list cannot be 0");
        }

        std::copy(list.begin(), list.end(), _data);
    }

    template <typename Operator>
    NdArray(const NdArraySlice<T, Dim, Operator> &array_slice)
        : NdArrayBase<T, Dim, NdArray<T, Dim>>(array_slice.shape), _data(new T[this->size()]) {
        std::array<index_t, Dim> indices;
        const index_t size = this->size();
        for (index_t i = 0; i < size; ++i) {
            util::unravel_index<Dim>(i, this->shape, indices);
            this->_data[i] = array_slice[indices];
        }
    }

    ~NdArray() {
        delete[] _data;
    }

    NdArray(const NdArray<T, Dim> &other)
        : NdArrayBase<T, Dim, NdArray<T, Dim>>(other.shape), _data(new T[other.shape.size()]) {
        std::copy(other._data, other._data + other.shape.size(), this->_data);
    }

    NdArray(NdArray<T, Dim> &&other) : NdArrayBase<T, Dim, NdArray<T, Dim>>(other.shape), _data(other._data) {
        other._data = nullptr;
    }

    NdArray<T, Dim> &operator=(const NdArray<T, Dim> &other) {
        if (this != &other) {
            delete[] this->_data;
            this->shape = other.shape;
            this->_data = new T[other.shape.size()];
            std::copy(other._data, other._data + other.shape.size(), this->_data);
        }

        return *this;
    }

    NdArray<T, Dim> &operator=(NdArray<T, Dim> &&other) {
        if (this != &other) {
            delete[] this->_data;
            this->shape = other.shape;
            this->_data = other._data;
            other._data = nullptr;
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

        return _data[index];
    }

    const T &operator[](const std::array<index_t, Dim> &indices) const {
        std::array<index_t, Dim> normalized_indices = util::normalize_indices(this->shape, indices);

        index_t index = 0;
        for (std::size_t i = 0; i < Dim; ++i) {
            index += normalized_indices[i] * this->shape.partial[i];
        }

        return _data[index];
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

    /* Assignment *****************************************************************************************************/

    NdArray<T, Dim> &operator=(const T &val) {
        this->fill(val);
        return *this;
    }

    /* Method *********************************************************************************************************/

    bool all(void) const {
        return std::all_of(this->_data, this->_data + this->size(), [](const T &val) { return val; });
    }

    bool any(void) const {
        return std::any_of(this->_data, this->_data + this->size(), [](const T &val) { return val; });
    }

    template <typename U>
    NdArray<U, Dim> as_type(void) const {
        NdArray<U, Dim> result(this->shape);

        std::transform(this->_data, this->_data + this->shape.size(), result._data,
                       [](const T &val) { return static_cast<U>(val); });

        return result;
    }

    T *data(void) {
        return this->_data;
    }

    const T *data(void) const {
        return this->_data;
    }

    void fill(const T &val) {
        std::fill(this->_data, this->_data + this->size(), val);
    }

    T &item(index_t index) {
        index_t size = this->size();

        if (index < -static_cast<index_t>(size) || index >= static_cast<index_t>(size)) {
            throw std::out_of_range("Index " + std::to_string(index) + " is out of bounds for size " +
                                    std::to_string(size));
        }

        if (index < 0) {
            index += size;
        }

        return this->_data[index];
    }

    const T &item(index_t index) const {
        index_t size = this->size();

        if (index < -static_cast<index_t>(size) || index >= static_cast<index_t>(size)) {
            throw std::out_of_range("Index " + std::to_string(index) + " is out of bounds for size " +
                                    std::to_string(size));
        }

        if (index < 0) {
            index += size;
        }

        return this->_data[index];
    }

private:
    template <typename, std::size_t>
    friend class NdArray;
    template <typename, std::size_t, typename>
    friend class NdArraySlice;

    T *_data;
};

template <typename T>
class NdArray<T, 0>;

}  // namespace ndarray

#endif
