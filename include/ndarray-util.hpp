#ifndef NDARRAY_UTIL_HPP
#define NDARRAY_UTIL_HPP

#include <array>
#include <type_traits>
#include <typeinfo>
#ifndef _MSC_VER
#include <cxxabi.h>
#endif

namespace ndarray {

template <std::size_t Dim>
class Size;

class Slice;

namespace util {

template <typename T>
constexpr bool is_index_type = std::is_integral_v<std::remove_reference_t<T>>;

template <typename T>
constexpr bool is_slice_type = std::is_same_v<std::remove_reference_t<T>, Slice>;

template <typename T>
constexpr bool is_index_slice_type = is_index_type<T> || is_slice_type<T>;

template <typename... Args>
constexpr std::size_t count_slice_type = (is_slice_type<Args> + ...);

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
std::array<index_t, Dim> normalize_indices(const Size<Dim> &shape, const std::array<index_t, Dim> &indices) {
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
void normalize_indices_slices(const Size<NIndices + NSlices> &shape,
                              const std::array<bool, NIndices + NSlices> &is_slice_axis,
                              std::array<index_t, NIndices> &indices, std::array<Slice, NSlices> &slices) {
    for (std::size_t i = 0, j = 0, k = 0; i < NIndices + NSlices; i++) {
        if (is_slice_axis[i]) {
            slices[j].normalize(shape[i]);
            ++j;
        } else {
            if (indices[k] < 0 || indices[k] >= shape[i]) {
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
