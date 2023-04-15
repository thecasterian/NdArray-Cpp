#ifndef NDARRAY_SLICE_HPP
#define NDARRAY_SLICE_HPP

#include <algorithm>
#include <array>
#include <iostream>
#include <limits>

#include "ndarray-definition.hpp"
#include "ndarray-expr.hpp"
#include "ndarray-size.hpp"

namespace ndarray {

template <typename T, std::size_t Dim, typename Operand>
class NdArraySlice;

class Slice {
public:
    static constexpr index_t none = std::numeric_limits<index_t>::max();

    Slice() : start(0), stop(none), step(1) {}
    explicit Slice(index_t stop) : start(0), stop(stop), step(1) {}
    explicit Slice(index_t start, index_t stop) : start(start), stop(stop), step(1) {}
    explicit Slice(index_t start, index_t stop, index_t step) : start(start), stop(stop), step(step) {
        if (step == 0) {
            throw std::invalid_argument("Step cannot be 0");
        }
    }

    operator std::string() const {
        std::string start = this->start == none ? "none" : std::to_string(this->start);
        std::string stop = this->stop == none ? "none" : std::to_string(this->stop);

        return "Slice(" + start + ", " + stop + ", " + std::to_string(this->step) + ")";
    }

private:
    template <typename, std::size_t, typename>
    friend class NdArraySlice;

    void apply_size(index_t size) {
        /* Make positive. */
        if (this->start != none && this->start < 0) {
            this->start += size;
        }
        if (this->stop != none && this->stop < 0) {
            this->stop += size;
        }

        /* Remove none. */
        if (this->step > 0) {
            this->start = this->start == none ? 0 : this->start;
            this->stop = this->stop == none ? size : this->stop;
        } else {
            this->start = this->start == none ? size - 1 : this->start;
            this->stop = this->stop == none ? -1 : this->stop;
        }

        /* Clip start and stop. */
        if (this->step > 0) {
            this->start = std::clamp<index_t>(this->start, 0, size);
            this->stop = std::clamp<index_t>(this->stop, this->start, size);
        }
        if (this->step < 0) {
            this->start = std::clamp<index_t>(this->start, -1, size - 1);
            this->stop = std::clamp<index_t>(this->stop, this->start, size - 1);
        }
    }

    index_t len(void) const {
        if (this->start == this->stop) {
            return 0;
        }

        if (this->step > 0) {
            return (this->stop - this->start - 1) / this->step + 1;
        } else {
            return (this->start - this->stop - 1) / (-this->step) + 1;
        }
    }

    index_t start;
    index_t stop;
    index_t step;
};

template <typename T, std::size_t Dim, typename Operand>
class NdArraySlice : public NdArrayExpr<T, Dim, NdArraySlice<T, Dim, Operand>> {
public:
    using NdArrayExpr<T, Dim, NdArraySlice<T, Dim, Operand>>::operator[];

    NdArraySlice(Operand &operand, const std::array<bool, Operand::dim> &is_slice_axis,
                 const std::array<index_t, Operand::dim - Dim> &indices, const std::array<Slice, Dim> &slices)
        : operand(operand),
          is_slice_axis(is_slice_axis),
          indices(indices),
          slices(this->apply_size(operand.shape, is_slice_axis, slices)) {
        for (std::size_t i = 0; i < Dim; ++i) {
            this->shape.size[i] = this->slices[i].len();
        }
    }

    NdArraySlice(const NdArraySlice &other) = delete;
    NdArraySlice(NdArraySlice &&other) = delete;
    NdArraySlice &operator=(const NdArraySlice &other) = delete;
    NdArraySlice &operator=(NdArraySlice &&other) = delete;

    template <typename... Args>
        requires(sizeof...(Args) == Dim && (util::is_index_type<Args> && ...))
    T &operator[](Args... args) {
        std::array<index_t, Dim> indices = {static_cast<index_t>(args)...};
        return this->operator[](indices);
    }

    template <typename... Args>
        requires(sizeof...(Args) == Dim && (util::is_index_type<Args> && ...))
    T operator[](Args... args) const {
        std::array<index_t, Dim> indices = {static_cast<index_t>(args)...};
        return this->operator[](indices);
    }

    T &operator[](const std::array<index_t, Dim> &indices) {
        this->validate_indices(indices);

        std::array<index_t, Operand::dim> operand_indices;
        for (std::size_t i = 0, j = 0, k = 0; i < Operand::dim; ++i) {
            if (this->is_slice_axis[i]) {
                operand_indices[i] = this->slices[j].start + indices[j] * this->slices[j].step;
                ++j;
            } else {
                operand_indices[i] = this->indices[k];
                ++k;
            }
        }

        return this->operand.operator[](operand_indices);
    }

    T operator[](const std::array<index_t, Dim> &indices) const {
        this->validate_indices(indices);

        std::array<index_t, Operand::dim> operand_indices;
        for (std::size_t i = 0, j = 0, k = 0; i < Operand::dim; ++i) {
            if (this->is_slice_axis[i]) {
                operand_indices[i] = this->slices[j].start + indices[j] * this->slices[j].step;
                ++j;
            } else {
                operand_indices[i] = this->indices[k];
                ++k;
            }
        }

        return this->operand.operator[](operand_indices);
    }

    template <typename Derived>
    NdArraySlice<T, Dim, Operand> &operator=(const NdArrayExpr<T, Dim, Derived> &other) {
        // TODO: Implement.
        return *this;
    }

    NdArraySlice<T, Dim, Operand> &operator=(const T &val) {
        // TODO: Implement.
        return *this;
    }

private:
    std::array<Slice, Dim> apply_size(const Size<Operand::dim> &size, const std::array<bool, Operand::dim> &is_slice_axis,
                                      const std::array<Slice, Dim> &slices) {
        std::array<Slice, Dim> res;
        for (std::size_t i = 0, j = 0; i < Operand::dim; ++i) {
            if (is_slice_axis[i]) {
                res[j] = slices[j];
                res[j].apply_size(size.size[i]);
                ++j;
            }
        }
        return res;
    }

    Operand &operand;
    const std::array<bool, Operand::dim> is_slice_axis;
    const std::array<index_t, Operand::dim - Dim> indices;
    const std::array<Slice, Dim> slices;
};

std::ostream &operator<<(std::ostream &os, const Slice &slice) {
    os << static_cast<std::string>(slice);
    return os;
}

}  // namespace ndarray

#endif
