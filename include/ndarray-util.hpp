#ifndef NDARRAY_UTIL_HPP
#define NDARRAY_UTIL_HPP

#include <type_traits>

namespace nda {

class Slice;

namespace util {

template <typename T>
constexpr bool is_index_type = std::is_integral_v<T>;

template <typename T>
constexpr bool is_index_slice_type = std::is_integral_v<T> || std::is_same_v<T, Slice>;

}  // namespace util

}  // namespace nda

#endif
