#ifndef NDARRAY_UTIL_HPP
#define NDARRAY_UTIL_HPP

#include <type_traits>

namespace ndarray {

class Slice;

namespace util {

template <typename T>
constexpr bool is_index_type = std::is_integral_v<T>;

template <typename T>
constexpr bool is_slice_type = std::is_same_v<T, Slice>;

template <typename T>
constexpr bool is_index_slice_type = is_index_type<T> || is_slice_type<T>;

template <typename... Args>
constexpr std::size_t count_slice_type = (is_slice_type<Args> + ...);

}  // namespace util

}  // namespace ndarray

#endif
