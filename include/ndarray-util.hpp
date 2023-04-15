#ifndef NDARRAY_UTIL_HPP
#define NDARRAY_UTIL_HPP

#include <array>
#include <type_traits>

namespace ndarray {

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

template <std::size_t Nindices, std::size_t Nslices, typename T, typename... Args>
void separate_index_slice(typename std::array<index_t, Nindices>::iterator indices_it,
                          typename std::array<Slice, Nslices>::iterator slices_it, T arg, Args... args) {
    if constexpr (is_index_type<T>) {
        *indices_it = arg;
        if constexpr (sizeof...(Args) > 0) {
            separate_index_slice<Nindices, Nslices>(indices_it + 1, slices_it, args...);
        }
    } else if constexpr (is_slice_type<T>) {
        *slices_it = arg;
        if constexpr (sizeof...(Args) > 0) {
            separate_index_slice<Nindices, Nslices>(indices_it, slices_it + 1, args...);
        }
    }
}

}  // namespace util

}  // namespace ndarray

#endif
