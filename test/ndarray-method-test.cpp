#include <gtest/gtest.h>

#include "../include/ndarray.hpp"

using namespace ndarray;

TEST(NdArrayMethodTest, All) {
    const NdArray<bool, 2> a = {{true, true, true}, {true, true, true}};
    const NdArray<bool, 3> b = {{{true, true, true}, {true, true, true}},
                                {{true, true, true}, {false, false, false}},
                                {{true, true, true}, {true, true, true}}};
    const NdArray<int, 1> c = {0, 0, 0, 0};

    ASSERT_TRUE(a.all());
    ASSERT_FALSE(b.all());
    ASSERT_TRUE((b[0, ":", ":"]).all());
    ASSERT_TRUE((b[":", 0, ":"]).all());
    ASSERT_FALSE((b[1, 1, ":"]).all());
    ASSERT_FALSE((b[":", 1, 1]).all());
    ASSERT_FALSE(c.all());
}

TEST(NdArrayMethodTest, Any) {
    const NdArray<bool, 2> a = {{true, true, true}, {true, true, true}};
    const NdArray<bool, 3> b = {{{true, true, true}, {true, true, true}},
                                {{true, true, true}, {false, false, false}},
                                {{true, true, true}, {true, true, true}}};
    const NdArray<int, 1> c = {0, 0, 0, 0};

    ASSERT_TRUE(a.any());
    ASSERT_TRUE(b.any());
    ASSERT_TRUE((b[0, ":", ":"]).any());
    ASSERT_TRUE((b[":", 0, ":"]).any());
    ASSERT_FALSE((b[1, 1, ":"]).any());
    ASSERT_TRUE((b[":", 1, 1]).any());
    ASSERT_FALSE(c.any());
}

TEST(NdArrayMethodTest, AsType) {
    const NdArray<int, 2> a = {{1, 2, 3}, {4, 5, 6}};
    const NdArray<long, 2> la = {{2, 3}, {5, 6}};

    const NdArray<Slice, 1> b = {Slice(0, 3, 1), Slice(Slice::none, Slice::none, Slice::none)};
    const NdArray<std::string, 1> sb = {"Slice(0, 3, 1)", "Slice(none, none, none)"};

    auto c = b.as_type<std::string>();
    std::cout << c[0] << ' ' << c[1] << std::endl;

    ASSERT_TRUE((a[":", "1:"].as_type<long>() == la).all());
    ASSERT_TRUE((b.as_type<std::string>() == sb).all());
}

TEST(NdArrayMethodTest, Fill) {
    NdArray<int, 3> a = {{{0, 1, 2}, {3, 4, 5}, {6, 7, 8}},
                         {{9, 10, 11}, {12, 13, 14}, {15, 16, 17}},
                         {{18, 19, 20}, {21, 22, 23}, {24, 25, 26}}};
    const NdArray<int, 3> fa = {{{0, 1, 2}, {3, -1, -1}, {6, -1, -1}},
                                {{9, 10, 11}, {12, -1, -1}, {15, -1, -1}},
                                {{18, 19, 20}, {21, -1, -1}, {24, -1, -1}}};

    NdArray<float, 2> b = {{1.0, 2.0}, {3.0, 4.0}};
    const NdArray<float, 2> fb = {{-1.0, -1.0}, {-1.0, -1.0}};

    a[":", "1:3", "1:3"].fill(-1);
    b.fill(-1);

    EXPECT_TRUE((a == fa).all());
    EXPECT_TRUE((b == fb).all());
}

TEST(NdArrayMethodTest, Flatten) {
    const NdArray<int, 2> a = {{0, 1, 2, 3}, {4, 5, 6, 7}};
    const NdArray<int, 1> fa = {0, 1, 2, 3, 4, 5, 6, 7};
    const NdArray<int, 3> b = {{{0, 1, 2}, {3, 4, 5}, {6, 7, 8}},
                               {{9, 10, 11}, {12, 13, 14}, {15, 16, 17}},
                               {{18, 19, 20}, {21, 22, 23}, {24, 25, 26}}};
    const NdArray<int, 1> fb = {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13,
                                14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26};

    ASSERT_TRUE((a.flatten() == fa).all());
    ASSERT_TRUE((b.flatten() == fb).all());
}

TEST(NdArrayMethodTest, Reshape) {
    const NdArray<int, 1> a = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
    const NdArray<int, 2> ra1 = {{0, 1, 2, 3}, {4, 5, 6, 7}, {8, 9, 10, 11}};
    const NdArray<int, 3> ra2 = {{{0, 1}, {2, 3}}, {{4, 5}, {6, 7}}, {{8, 9}, {10, 11}}};

    ASSERT_TRUE((a.reshape(Shape<2>({3, 4})) == ra1).all());
    ASSERT_TRUE((a.reshape(Shape<3>({3, 2, 2})) == ra2).all());
    EXPECT_ANY_THROW(a.reshape(Shape<2>({5, 5})));
}

TEST(NdArrayMethodTest, ToString1) {
    const NdArray<int, 2> a = {{1, 2, 3}, {4, 5, 6}};
    const std::string s = "NdArray({{1, 2, 3}, {4, 5, 6}})";

    ASSERT_EQ(a.to_string(), s);
}

TEST(NdArrayMethodTest, ToString2) {
    const NdArray<int, 3> a = {{{0, 1, 2}, {3, 4, 5}, {6, 7, 8}},
                               {{9, 10, 11}, {12, 13, 14}, {15, 16, 17}},
                               {{18, 19, 20}, {21, 22, 23}, {24, 25, 26}}};
    const std::string s = "NdArray({{{4, 5}, {7, 8}}, {{13, 14}, {16, 17}}, {{22, 23}, {25, 26}}})";

    ASSERT_EQ((a[":", "1:3", "1:3"]).to_string(), s);
}

TEST(NdArrayMethodTest, ToVector1) {
    const NdArray<int, 2> a = {{1, 2, 3}, {4, 5, 6}};
    const std::vector<std::vector<int>> v = {{1, 2, 3}, {4, 5, 6}};

    ASSERT_EQ(a.to_vector(), v);
}

TEST(NdArrayMethodTest, ToVector2) {
    const NdArray<int, 3> a = {{{0, 1, 2}, {3, 4, 5}, {6, 7, 8}},
                               {{9, 10, 11}, {12, 13, 14}, {15, 16, 17}},
                               {{18, 19, 20}, {21, 22, 23}, {24, 25, 26}}};
    const std::vector<std::vector<std::vector<int>>> v = {{{4, 5}, {7, 8}}, {{13, 14}, {16, 17}}, {{22, 23}, {25, 26}}};

    ASSERT_EQ((a[":", "1:3", "1:3"]).to_vector(), v);
}
