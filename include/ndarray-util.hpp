#ifndef NDARRAY_UTIL_HPP
#define NDARRAY_UTIL_HPP

#include <array>
#include <concepts>
#include <type_traits>
#include <typeinfo>
#ifndef _MSC_VER
#include <cxxabi.h>
#endif

namespace ndarray {

template <std::size_t Dim>
class Shape;

class Slice;

namespace util {

template <typename T>
constexpr bool is_index_type = std::is_integral_v<std::remove_reference_t<T>>;

template <typename T>
constexpr bool is_slice_type =
    std::is_same_v<std::remove_reference_t<T>, Slice> || std::is_same_v<std::remove_reference_t<T>, const char *> ||
    std::is_same_v<std::remove_reference_t<T>, std::string>;

template <typename T>
constexpr bool is_index_slice_type = is_index_type<T> || is_slice_type<T>;

template <typename... Args>
constexpr std::size_t count_slice_type = (is_slice_type<Args> + ...);

inline index_t to_index_t(const std::string &str) {
    std::string::size_type sz;
    index_t ret = std::stoll(str, &sz);
    if (sz != str.size()) {
        throw std::invalid_argument("Invalid index: " + str);
    }
    return ret;
}

template <std::size_t OperandDim, std::size_t Dim>
std::array<index_t, Dim> pick_slice_axes(const std::array<bool, OperandDim> &is_slice_axis) {
    std::array<index_t, Dim> slice_axes;
    for (std::size_t i = 0, j = 0; i < OperandDim; ++i) {
        if (is_slice_axis[i]) {
            slice_axes[j] = i;
            ++j;
        }
    }
    return slice_axes;
}

template <std::size_t NIndices, std::size_t NSlices, typename T, typename... Args>
void separate_index_slice(typename std::array<index_t, NIndices>::iterator indices_it,
                          typename std::array<Slice, NSlices>::iterator slices_it, T arg, Args... args) {
    if constexpr (is_index_type<T>) {
        *indices_it = arg;
        if constexpr (sizeof...(Args) > 0) {
            separate_index_slice<NIndices, NSlices>(indices_it + 1, slices_it, args...);
        }
    } else if constexpr (is_slice_type<T>) {
        *slices_it = arg;
        if constexpr (sizeof...(Args) > 0) {
            separate_index_slice<NIndices, NSlices>(indices_it, slices_it + 1, args...);
        }
    }
}

template <std::size_t Dim>
Shape<Dim> slices_to_shape(const std::array<Slice, Dim> &slices) {
    std::array<index_t, Dim> shape;
    for (std::size_t i = 0; i < Dim; ++i) {
        shape[i] = slices[i].len();
    }
    return Shape<Dim>(shape);
}

template <std::size_t Dim>
std::array<index_t, Dim> normalize_indices(const Shape<Dim> &shape, const std::array<index_t, Dim> &indices) {
    std::array<index_t, Dim> normalized_indices;
    for (std::size_t i = 0; i < Dim; ++i) {
        if (indices[i] < -shape[i] || indices[i] >= shape[i]) {
            throw std::out_of_range("Index " + std::to_string(indices[i]) + " is out of range for axis " +
                                    std::to_string(i) + " with size " + std::to_string(shape[i]));
        }

        normalized_indices[i] = indices[i] >= 0 ? indices[i] : indices[i] + shape[i];
    }
    return normalized_indices;
}

template <std::size_t NIndices, std::size_t NSlices>
void normalize_indices_slices(const Shape<NIndices + NSlices> &shape,
                              const std::array<bool, NIndices + NSlices> &is_slice_axis,
                              std::array<index_t, NIndices> &indices, std::array<Slice, NSlices> &slices) {
    for (std::size_t i = 0, j = 0, k = 0; i < NIndices + NSlices; i++) {
        if (is_slice_axis[i]) {
            slices[j].normalize(shape[i]);
            ++j;
        } else {
            if (indices[k] < -shape[i] || indices[k] >= shape[i]) {
                throw std::out_of_range("Index " + std::to_string(indices[k]) + " is out of range for axis " +
                                        std::to_string(i) + " with size " + std::to_string(shape[i]));
            }
            if (indices[k] < 0) {
                indices[k] += shape[i];
            }
            ++k;
        }
    }
}

template <std::size_t Dim>
void unravel_index(index_t index, const Shape<Dim> &shape, std::array<index_t, Dim> &indices) {
    for (index_t i = static_cast<index_t>(Dim) - 1; i >= 0; --i) {
        indices[i] = index % shape[i];
        index /= shape[i];
    }
}

template <typename T1, typename T2>
using eq_t = decltype(std::declval<T1>() == std::declval<T2>());

template <typename T1, typename T2>
using neq_t = decltype(std::declval<T1>() != std::declval<T2>());

template <typename T1, typename T2>
using lt_t = decltype(std::declval<T1>() < std::declval<T2>());

template <typename T1, typename T2>
using gt_t = decltype(std::declval<T1>() > std::declval<T2>());

template <typename T1, typename T2>
using le_t = decltype(std::declval<T1>() <= std::declval<T2>());

template <typename T1, typename T2>
using ge_t = decltype(std::declval<T1>() >= std::declval<T2>());

template <typename T1, typename T2>
using add_t = decltype(std::declval<T1>() + std::declval<T2>());

template <typename T1, typename T2>
using sub_t = decltype(std::declval<T1>() - std::declval<T2>());

template <typename T1, typename T2>
using mul_t = decltype(std::declval<T1>() * std::declval<T2>());

template <typename T1, typename T2>
using div_t = decltype(std::declval<T1>() / std::declval<T2>());

template <typename T1, typename T2>
using mod_t = decltype(std::declval<T1>() % std::declval<T2>());

template <typename T, std::size_t Dim>
class NestedVector {
public:
    using type = std::vector<typename NestedVector<T, Dim - 1>::type>;
};

template <typename T>
class NestedVector<T, 1> {
public:
    using type = std::vector<T>;
};

template <typename T, std::size_t Dim>
using nested_vector_t = typename NestedVector<T, Dim>::type;

template <typename T>
std::string type_name(void) {
    using RemoveRefT = std::remove_reference_t<T>;

    std::string name;

#ifndef _MSC_VER
    char *realname = abi::__cxa_demangle(typeid(RemoveRefT).name(), nullptr, nullptr, nullptr);
    name = realname;
    free(realname);
#else
    name = typeid(RemoveRefT).name();
#endif

    if (std::is_const_v<RemoveRefT>) {
        name += " const";
    }
    if (std::is_volatile_v<RemoveRefT>) {
        name += " volatile";
    }
    if (std::is_lvalue_reference_v<T>) {
        name += "&";
    } else if (std::is_rvalue_reference_v<T>) {
        name += "&&";
    }

    return name;
}

}  // namespace util

}  // namespace ndarray

#endif
