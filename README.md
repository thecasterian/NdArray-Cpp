# NdArray-Cpp
Multidimensional array in C++, inspired by NumPy.

# Requirements
- C++23

## Features
### NdArray
`ndarray::NdArray` is a class template that represents a multidimensional array. Its data type and dimension is specified in compile time. It provides a similar interface to NumPy's `ndarray` class.

```cpp
// Create a 3-dimensional array of int.
ndarray::NdArray<int, 3> a = {{{0, 1, 2}, {3, 4, 5}}, {{6, 7, 8}, {9, 10, 11}}};
// Print its shape.
std::cout << a.shape() << std::endl;    // Shape(2, 2, 3)
// Print itself.
std::cout << a << std::endl;            // NdArray({{{0, 1, 2}, {3, 4, 5}}, {{6, 7, 8}, {9, 10, 11}}})
```

### Indexing
`ndarray::NdArray` supports indexing to access its elements. It can be done by using `operator[]` with multiple arguments.

```cpp
std::cout << a[0, 0, 0] << std::endl;   // 0
std::cout << a[1, 0, 2] << std::endl;   // 8
// a[2, 0, 0]                           // Out of range exception.
```

Also a negative index is supported.

```cpp
std::cout << a[-1, -1, -1] << std::endl;    // 11
// a[-1, -1, -4]                            // Out of range exception.
```

Indexing can be used to modify the element.

```cpp
ndarray::NdArray<int, 3> b = a;
b[0, 0, 0] = -1;
std::cout << b << std::endl;            // NdArray({{{-1, 1, 2}, {3, 4, 5}}, {{6, 7, 8}, {9, 10, 11}}})
```

### Slicing
`ndarray::NdArray` supports slicing to access its elements. It can be done by using `operator[]` with `ndarray::Slice` as an argument.

```cpp
// Get the subarray of a.
ndarray::NdArray<int, 2> sub_a1 = a[ndarray::Slice(0, 2), 1, ndarray::Slice(0, 3, 2)];
std::cout << sub_a1 << std::endl;       // NdArray({{3, 5}, {9, 11}})

ndarray::NdArray<int, 1> sub_a2 = a[0, 0, ndarray::Slice(2, 0, -1)];
std::cout << sub_a2 << std::endl;       // NdArray({2, 1})
```

If the number of arguments is less than the number of dimensions, the remaining dimensions are implicitly sliced.

```cpp
ndarray::NdArray<int, 2> sub_a3 = a[0];
std::cout << sub_a3 << std::endl;       // NdArray({{0, 1, 2}, {3, 4, 5}})
```

A python-style slice as a string is also supported.
```cpp
std::cout << a[":", 1, "::2"] << std::endl;     // NdArray({{3, 5}, {9, 11}})
std::cout << a[0, 0, ":0:-1"] << std::endl;     // NdArray({2, 1})
```

A sliced array is a view of the original array, so it can be used to modify the original array.

```cpp
ndarray::NdArray<int, 3> c = a;
c[":", 1, "::2"] = ndarray::NdArray<int, 2>({{-1, -2}, {-3, -4}});
std::cout << c << std::endl;            // NdArray({{{0, 1, 2}, {-1, 4, -2}}, {{6, 7, 8}, {-3, 10, -4}}})
```

### Operations

Several arithmetic operations and utility methods/functions are provided.

```cpp
ndarray::NdArray<int, 2> x = {{0, 1, 2}, {3, 4, 5}};
ndarray::NdArray<int, 2> y = {{6, 7, 8}, {9, 10, 11}};

// Add two arrays.
std::cout << x + y << std::endl;            // NdArray({{6, 8, 10}, {12, 14, 16}})
// Comapre two arrays.
std::cout << (x == y).any() << std::endl;   // 0
std::cout << (x < y).all() << std::endl;    // 1

// Reshape an array.
std::cout << x.reshape<2>({3, 2}) << std::endl;    // NdArray({{0, 1}, {2, 3}, {4, 5}})
```
