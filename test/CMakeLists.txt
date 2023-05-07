add_executable(ndarray-method-test ndarray-method-test.cpp)
target_link_libraries(ndarray-method-test GTest::gtest_main)

add_executable(ndarray-slice-test ndarray-slice-test.cpp)
target_link_libraries(ndarray-slice-test GTest::gtest_main)

add_executable(ndarray-op-test ndarray-op-test.cpp)
target_link_libraries(ndarray-op-test GTest::gtest_main)

include(GoogleTest)
gtest_discover_tests(ndarray-method-test)
gtest_discover_tests(ndarray-slice-test)
gtest_discover_tests(ndarray-op-test)
