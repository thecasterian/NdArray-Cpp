#include <gtest/gtest.h>

#include "../include/ndarray.hpp"

using namespace ndarray;

TEST(NdArrayBaseTest, ToString1) {
    const NdArray<int, 2> a = {{1, 2, 3}, {4, 5, 6}};
    const std::string s = "NdArray({{1, 2, 3}, {4, 5, 6}})";

    ASSERT_EQ(a.to_string(), s);
}

TEST(NdArrayBaseTest, ToString2) {
    const NdArray<int, 3> a = {{{0, 1, 2}, {3, 4, 5}, {6, 7, 8}},
                               {{9, 10, 11}, {12, 13, 14}, {15, 16, 17}},
                               {{18, 19, 20}, {21, 22, 23}, {24, 25, 26}}};
    const std::string s = "NdArray({{{4, 5}, {7, 8}}, {{13, 14}, {16, 17}}, {{22, 23}, {25, 26}}})";

    ASSERT_EQ((a[":", "1:3", "1:3"]).to_string(), s);
}

TEST(NdArrayBaseTest, ToVector1) {
    const NdArray<int, 2> a = {{1, 2, 3}, {4, 5, 6}};
    const std::vector<std::vector<int>> v = {{1, 2, 3}, {4, 5, 6}};

    ASSERT_EQ(a.to_vector(), v);
}

TEST(NdArrayBaseTest, ToVector2) {
    const NdArray<int, 3> a = {{{0, 1, 2}, {3, 4, 5}, {6, 7, 8}},
                               {{9, 10, 11}, {12, 13, 14}, {15, 16, 17}},
                               {{18, 19, 20}, {21, 22, 23}, {24, 25, 26}}};
    const std::vector<std::vector<std::vector<int>>> v = {{{4, 5}, {7, 8}}, {{13, 14}, {16, 17}}, {{22, 23}, {25, 26}}};

    ASSERT_EQ((a[":", "1:3", "1:3"]).to_vector(), v);
}
