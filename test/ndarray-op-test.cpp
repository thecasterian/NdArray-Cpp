#include <gtest/gtest.h>

#include "../include/ndarray.hpp"

using namespace ndarray;

TEST(BinaryArithmeticOpTest, Add) {
    const NdArray<int, 2> a = {{1, 2, 3}, {4, 5, 6}};
    const NdArray<int, 3> b = {
        {{1, 2, 3, 4}, {5, 6, 7, 8}}, {{9, 10, 11, 12}, {13, 14, 15, 16}}, {{17, 18, 19, 20}, {21, 22, 23, 24}}};
    const NdArray<int, 2> c = {{11, 13, 15}, {18, 20, 22}};

    const auto d = a + b[1, ":", "1:"];

    ASSERT_TRUE((c == d).all());
}

TEST(BinaryArithmeticOpTest, AddString) {
    const NdArray<std::string, 2> a = {{"a", "b", "c"}, {"d", "e", "f"}};
    const NdArray<std::string, 2> b = {{"A", "B", "C"}, {"D", "E", "F"}};
    const NdArray<std::string, 2> c = {{"aA", "bB", "cC"}, {"dD", "eE", "fF"}};

    const auto d = a + b;

    ASSERT_TRUE((c == d).all());
}
