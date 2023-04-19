#include "ndarray-slice.hpp"

#include <gtest/gtest.h>

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
