enable_testing()

include_directories(${CMAKE_SOURCE_DIR}/include)

add_executable(ndarray-slice-test ndarray-slice-test.cpp)
target_link_libraries(ndarray-slice-test GTest::gtest_main)

include(GoogleTest)
gtest_discover_tests(ndarray-slice-test)
