#ifndef NDARRAY_SLICE_HPP
#define NDARRAY_SLICE_HPP

#include "ndarray-definition.hpp"

namespace nda {

class Slice {
public:
    Slice() : start(0), stop(0), step(1) {}
    Slice(index_t start, index_t stop, index_t step = 1) : start(start), stop(stop), step(step) {}

private:
    index_t start;
    index_t stop;
    index_t step;
};

}

#endif
