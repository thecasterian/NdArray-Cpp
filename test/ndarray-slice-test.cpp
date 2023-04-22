#include <gtest/gtest.h>

#include "ndarray.hpp"

using namespace ndarray;

TEST(SliceTest, Constructor) {
    Slice s1;
    Slice s2(5);
    Slice s3(2, 8);
    Slice s4(13, 3, -2);

    EXPECT_EQ(s1.start, 0);
    EXPECT_EQ(s1.stop, Slice::none);
    EXPECT_EQ(s1.step, 1);

    EXPECT_EQ(s2.start, 0);
    EXPECT_EQ(s2.stop, 5);
    EXPECT_EQ(s2.step, 1);

    EXPECT_EQ(s3.start, 2);
    EXPECT_EQ(s3.stop, 8);
    EXPECT_EQ(s3.step, 1);

    EXPECT_EQ(s4.start, 13);
    EXPECT_EQ(s4.stop, 3);
    EXPECT_EQ(s4.step, -2);
}

TEST(SliceTest, FromString) {
    Slice s1(":");
    Slice s2("::");
    Slice s3("1:");
    Slice s4(":-2");
    Slice s5("-1::");
    Slice s6(":2:");
    Slice s7("::3");
    Slice s8("1:2:");
    Slice s9("1::-3");
    Slice s10(":-2:3");
    Slice s11("-1:2:-3");

    EXPECT_EQ(s1, Slice(Slice::none, Slice::none, 1));
    EXPECT_EQ(s2, Slice(Slice::none, Slice::none, 1));
    EXPECT_EQ(s3, Slice(1, Slice::none, 1));
    EXPECT_EQ(s4, Slice(Slice::none, -2, 1));
    EXPECT_EQ(s5, Slice(-1, Slice::none, 1));
    EXPECT_EQ(s6, Slice(Slice::none, 2, 1));
    EXPECT_EQ(s7, Slice(Slice::none, Slice::none, 3));
    EXPECT_EQ(s8, Slice(1, 2, 1));
    EXPECT_EQ(s9, Slice(1, Slice::none, -3));
    EXPECT_EQ(s10, Slice(Slice::none, -2, 3));
    EXPECT_EQ(s11, Slice(-1, 2, -3));
}

TEST(SliceTest, Normalize) {
    Slice s1(3, 15, 2);
    s1.normalize(10);
    EXPECT_EQ(s1, Slice(3, 11, 2));

    Slice s2(-4, Slice::none, -3);
    s2.normalize(13);
    EXPECT_EQ(s2, Slice(9, -3, -3));

    Slice s3(Slice::none, -2);
    s3.normalize(9);
    EXPECT_EQ(s3, Slice(0, 7, 1));

    Slice s4(Slice::none, Slice::none, -4);
    s4.normalize(12);
    EXPECT_EQ(s4, Slice(11, -1, -4));

    Slice s5(-1, 9);
    s5.normalize(5);
    EXPECT_EQ(s5, Slice(4, 5, 1));

    Slice s6(-3, 7, 2);
    s6.normalize(20);
    EXPECT_EQ(s6, Slice(17, 17, 2));

    Slice s7(5, 6, -1);
    s7.normalize(10);
    EXPECT_EQ(s7, Slice(5, 5, -1));
}

TEST(SliceTest, MergeIndex) {
    EXPECT_EQ(Slice(2, 18, 2) * 3, 8);
    EXPECT_EQ(Slice(2, 18, 2) * 7, 16);
    EXPECT_EQ(Slice(57, 7, -5) * 6, 27);
    EXPECT_EQ(Slice(57, 7, -5) * 1, 52);
}

TEST(SliceTest, MergeSlice) {
    EXPECT_EQ(Slice(2, 18, 2) * Slice(1, 10, 3), Slice(4, 22, 6));
    EXPECT_EQ(Slice(2, 18, 2) * Slice(7, -1, -1), Slice(16, 0, -2));
    EXPECT_EQ(Slice(57, 7, -5) * Slice(1, 11, 2), Slice(52, 2, -10));
    EXPECT_EQ(Slice(57, 7, -5) * Slice(6, -3, -3), Slice(27, 72, 15));
}

TEST(NdArraySliceTest, Assign1d) {
    NdArray<int, 1> a = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    const NdArray<int, 1> b = {-1, -2, -3};
    const NdArray<int, 1> c = {0, 1, -1, -2, -3, 0, 6, 0, 8, 0};

    a[Slice(2, 5)] = b;
    a[Slice(Slice::none, 4, -2)] = 0;

    EXPECT_EQ(a, c);
}

TEST(NdArraySliceTest, Assign2d) {
    NdArray<int, 3> a = {{{0, 1, 2}, {3, 4, 5}, {6, 7, 8}},
                         {{9, 10, 11}, {12, 13, 14}, {15, 16, 17}},
                         {{18, 19, 20}, {21, 22, 23}, {24, 25, 26}}};
    const NdArray<int, 2> b = {{-1, -2}, {-3, -4}};
    const NdArray<int, 3> c = {{{-1, -2, 2}, {3, 4, 5}, {-3, -4, 8}},
                               {{9, 0, 11}, {12, 0, 14}, {15, -2, -1}},
                               {{18, 0, 20}, {21, 0, 23}, {24, -4, -3}}};

    a[-3, ndarray::Slice(ndarray::Slice::none, ndarray::Slice::none, 2), ndarray::Slice(2)] = b;
    a[ndarray::Slice(1, ndarray::Slice::none), 2, ndarray::Slice(ndarray::Slice::none, 0, -1)] = b;
    a[ndarray::Slice(1, 3), ndarray::Slice(2), 1] = 0;

    EXPECT_EQ(a, c);
}

TEST(NdArraySliceTest, Assign1dString) {
    NdArray<int, 1> a = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    const NdArray<int, 1> b = {-1, -2, -3};
    const NdArray<int, 1> c = {0, 1, -1, -2, -3, 0, 6, 0, 8, 0};

    a["2:5"] = b;
    a[":4:-2"] = 0;

    EXPECT_EQ(a, c);
}

TEST(NdArraySliceTest, Assign2dString) {
    NdArray<int, 3> a = {{{0, 1, 2}, {3, 4, 5}, {6, 7, 8}},
                         {{9, 10, 11}, {12, 13, 14}, {15, 16, 17}},
                         {{18, 19, 20}, {21, 22, 23}, {24, 25, 26}}};
    const NdArray<int, 2> b = {{-1, -2}, {-3, -4}};
    const NdArray<int, 3> c = {{{-1, -2, 2}, {3, 4, 5}, {-3, -4, 8}},
                               {{9, 0, 11}, {12, 0, 14}, {15, -2, -1}},
                               {{18, 0, 20}, {21, 0, 23}, {24, -4, -3}}};

    a[-3, "::2", ":2"] = b;
    a["1:", 2, ":0:-1"] = b;
    a["1:3", ":2", 1] = 0;

    EXPECT_EQ(a, c);
}

TEST(NdArraySliceTest, NdArrayCast1d) {
    const NdArray<int, 1> a = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    const NdArray<int, 1> b = a["2:8"];
    const NdArray<int, 1> c = {2, 3, 4, 5, 6, 7};

    EXPECT_EQ(b, c);
}

TEST(NdArraySliceTest, NdArrayCast2d) {
    const NdArray<int, 3> a = {{{0, 1, 2}, {3, 4, 5}, {6, 7, 8}},
                               {{9, 10, 11}, {12, 13, 14}, {15, 16, 17}},
                               {{18, 19, 20}, {21, 22, 23}, {24, 25, 26}}};
    const NdArray<int, 3> b = a[":", "1:3", "1:3"];
    const NdArray<int, 3> c = {{{4, 5}, {7, 8}},
                               {{13, 14}, {16, 17}},
                               {{22, 23}, {25, 26}}};

    EXPECT_EQ(b, c);
}

TEST(NdArraySliceTest, Fill) {
    NdArray<int, 3> a = {{{0, 1, 2}, {3, 4, 5}, {6, 7, 8}},
                               {{9, 10, 11}, {12, 13, 14}, {15, 16, 17}},
                               {{18, 19, 20}, {21, 22, 23}, {24, 25, 26}}};
    const NdArray<int, 3> b = {{{0, 1, 2}, {3, -1, -1}, {6, -1, -1}},
    {{9, 10, 11}, {12, -1, -1}, {15, -1, -1}},
    {{18, 19, 20}, {21, -1, -1}, {24, -1, -1}}};

    a[":", "1:3", "1:3"].fill(-1);

    EXPECT_EQ(a, b);
}
