#ifndef NDARRAY_SLICE_HPP
#define NDARRAY_SLICE_HPP

#include <algorithm>
#include <array>
#include <iostream>
#include <limits>

#include "ndarray-definition.hpp"
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
            throw std::invalid_argument("Slice step cannot be zero");
        }
    }
    Slice(const std::string &str) {
        std::string::size_type pos1, pos2;

        pos1 = str.find(':');
        pos2 = str.find(':', pos1 + 1);

        if (pos1 == std::string::npos) {
            throw std::invalid_argument("Slice string must contain at least one colon");
        }

        if (pos1 == 0) {
            this->start = none;
        } else {
            this->start = std::stoll(str.substr(0, pos1));
        }

        if ((pos2 == std::string::npos && pos1 + 1 == str.size()) || (pos2 != std::string::npos && pos1 + 1 == pos2)) {
            this->stop = none;
        } else {
            this->stop = std::stoll(str.substr(pos1 + 1, pos2 - pos1 - 1));
        }

        if (pos2 == std::string::npos || pos2 + 1 == str.size()) {
            this->step = 1;
        } else {
            this->step = std::stoll(str.substr(pos2 + 1));
        }

        if (this->step == 0) {
            throw std::invalid_argument("Slice step cannot be zero");
        }
    }
    Slice(const char *str) : Slice(std::string(str)) {}

    index_t operator*(index_t index) const {
        return this->start + index * this->step;
    }

    Slice operator*(const Slice &other) const {
        Slice res;
        res.start = this->step * other.start + this->start;
        res.step = this->step * other.step;
        res.stop = res.start + res.step * other.len();
        return res;
    }

    void normalize(index_t size) {
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
            this->stop = std::clamp<index_t>(this->stop, -1, this->start);
        }

        /* Normalize stop. */
        this->stop = this->start + this->len() * this->step;
    }

    index_t len(void) const {
        if (this->step > 0) {
            return (this->stop - this->start + this->step - 1) / this->step;
        } else {
            return (this->start - this->stop - this->step - 1) / (-this->step);
        }
    }

    bool operator==(const Slice &other) const {
        return this->start == other.start && this->stop == other.stop && this->step == other.step;
    }

    bool operator!=(const Slice &other) const {
        return !(*this == other);
    }

    operator std::string() const {
        std::string start = this->start == none ? "none" : std::to_string(this->start);
        std::string stop = this->stop == none ? "none" : std::to_string(this->stop);

        return "Slice(" + start + ", " + stop + ", " + std::to_string(this->step) + ")";
    }

    index_t start;
    index_t stop;
    index_t step;
};

template <typename T, std::size_t Dim, typename Operand>
class NdArraySlice : public NdArrayBase<T, Dim> {
public:
    NdArraySlice(Operand &operand, const std::array<bool, Operand::dim> &is_slice_axis,
                 const std::array<index_t, Operand::dim - Dim> &indices, const std::array<Slice, Dim> &slices)
        : operand(operand),
          is_slice_axis(is_slice_axis),
          indices(indices),
          slices(slices),
          slice_axes(util::pick_slice_axes<Operand::dim, Dim>(is_slice_axis)) {
        for (std::size_t i = 0; i < Dim; ++i) {
            this->shape.size[i] = this->slices[i].len();
        }
    }

    NdArraySlice(const NdArraySlice &other) = delete;
    NdArraySlice(NdArraySlice &&other) = delete;
    NdArraySlice &operator=(const NdArraySlice &other) = delete;
    NdArraySlice &operator=(NdArraySlice &&other) = delete;

    /* Indexing *******************************************************************************************************/

    template <typename... Args>
        requires(sizeof...(Args) == Dim && (util::is_index_type<Args> && ...))
    T &operator[](Args... args) {
        std::array<index_t, Dim> indices = {static_cast<index_t>(args)...};
        return this->operator[](indices);
    }

    template <typename... Args>
        requires(sizeof...(Args) == Dim && (util::is_index_type<Args> && ...))
    const T &operator[](Args... args) const {
        std::array<index_t, Dim> indices = {static_cast<index_t>(args)...};
        return this->operator[](indices);
    }

    T &operator[](const std::array<index_t, Dim> &indices) {
        std::array<index_t, Dim> normalized_indices = util::normalize_indices(this->shape, indices);

        std::array<index_t, Operand::dim> operand_indices;
        for (std::size_t i = 0, j = 0, k = 0; i < Operand::dim; ++i) {
            if (this->is_slice_axis[i]) {
                operand_indices[i] = this->slices[j] * normalized_indices[j];
                ++j;
            } else {
                operand_indices[i] = this->indices[k];
                ++k;
            }
        }

        return this->operand.operator[](operand_indices);
    }

    const T &operator[](const std::array<index_t, Dim> &indices) const {
        std::array<index_t, Dim> normalized_indices = util::normalize_indices(this->shape, indices);

        std::array<index_t, Operand::dim> operand_indices;
        for (std::size_t i = 0, j = 0, k = 0; i < Operand::dim; ++i) {
            if (this->is_slice_axis[i]) {
                operand_indices[i] = this->slices[j] * normalized_indices[j];
                ++j;
            } else {
                operand_indices[i] = this->indices[k];
                ++k;
            }
        }

        return this->operand.operator[](operand_indices);
    }

    /* Slicing ********************************************************************************************************/

    template <typename... Args>
        requires(sizeof...(Args) <= Dim && (util::is_index_slice_type<Args> && ...) &&
                 !(sizeof...(Args) == Dim && (util::is_index_type<Args> && ...)))
    NdArraySlice<T, util::count_slice_type<Args...> + Dim - sizeof...(Args), Operand> operator[](Args... args) {
        static constexpr std::size_t NIndices = sizeof...(Args) - util::count_slice_type<Args...>;
        static constexpr std::size_t NSlices = util::count_slice_type<Args...> + Dim - sizeof...(Args);

        std::array<bool, Dim> is_slice_axis;
        std::array<index_t, NIndices> indices;
        std::array<Slice, NSlices> slices;

        index_t i = 0;
        ((is_slice_axis[i++] = util::is_slice_type<Args>), ...);
        for (std::size_t i = sizeof...(Args); i < Dim; ++i) {
            is_slice_axis[i] = true;
        }

        util::separate_index_slice<NIndices, NSlices, Args...>(indices.begin(), slices.begin(), args...);

        util::normalize_indices_slices<NIndices, NSlices>(this->shape, is_slice_axis, indices, slices);

        /* Merge with the current indices and slices. */
        std::array<bool, Operand::dim> is_slice_axis_new;
        std::array<index_t, Operand::dim - NSlices> indices_new;
        std::array<Slice, NSlices> slices_new;

        for (std::size_t i = 0, j = 0, k = 0, l = 0, m = 0, n = 0; i < Operand::dim; ++i) {
            if (this->is_slice_axis[i]) {
                if (is_slice_axis[j]) {
                    is_slice_axis_new[i] = true;
                    slices_new[l] = this->slices[j] * slices[l];
                    ++j;
                    ++l;
                } else {
                    is_slice_axis_new[i] = false;
                    indices_new[n] = this->slices[j] * indices[m];
                    ++j;
                    ++m;
                    ++n;
                }
            } else {
                is_slice_axis_new[i] = false;
                indices_new[n] = this->indices[k];
                ++k;
                ++n;
            }
        }

        return {this->operand, is_slice_axis_new, indices_new, slices_new};
    }

    template <typename... Args>
        requires(sizeof...(Args) <= Dim && (util::is_index_slice_type<Args> && ...) &&
                 !(sizeof...(Args) == Dim && (util::is_index_type<Args> && ...)))
    const NdArraySlice<T, util::count_slice_type<Args...> + Dim - sizeof...(Args), const Operand> operator[](
        Args... args) const {
        static constexpr std::size_t NIndices = sizeof...(Args) - util::count_slice_type<Args...>;
        static constexpr std::size_t NSlices = util::count_slice_type<Args...> + Dim - sizeof...(Args);

        std::array<bool, Dim> is_slice_axis;
        std::array<index_t, NIndices> indices;
        std::array<Slice, NSlices> slices;

        index_t i = 0;
        ((is_slice_axis[i++] = util::is_slice_type<Args>), ...);
        for (std::size_t i = sizeof...(Args); i < Dim; ++i) {
            is_slice_axis[i] = true;
        }

        util::separate_index_slice<NIndices, NSlices, Args...>(indices.begin(), slices.begin(), args...);

        util::normalize_indices_slices<NIndices, NSlices>(this->shape, is_slice_axis, indices, slices);

        /* Merge with the current indices and slices. */
        std::array<bool, Operand::dim> is_slice_axis_new;
        std::array<index_t, Operand::dim - NSlices> indices_new;
        std::array<Slice, NSlices> slices_new;

        for (std::size_t i = 0, j = 0, k = 0, l = 0, m = 0, n = 0; i < Operand::dim; ++i) {
            if (this->is_slice_axis[i]) {
                if (is_slice_axis[j]) {
                    is_slice_axis_new[i] = true;
                    slices_new[l] = this->slices[j] * slices[l];
                    ++j;
                    ++l;
                } else {
                    is_slice_axis_new[i] = false;
                    indices_new[n] = this->slices[j] * indices[m];
                    ++j;
                    ++m;
                    ++n;
                }
            } else {
                is_slice_axis_new[i] = false;
                indices_new[n] = this->indices[k];
                ++k;
                ++n;
            }
        }

        return {static_cast<const Operand &>(this->operand), is_slice_axis_new, indices_new, slices_new};
    }

    /* Assignment *****************************************************************************************************/

    NdArraySlice<T, Dim, Operand> &operator=(const NdArray<T, Dim> &other) {
        if (this->shape != other.shape) {
            throw std::invalid_argument("Could not broadcast input array from " + this->shape.to_string() + " to " +
                                        other.shape.to_string());
        }

        std::array<index_t, Operand::dim> operand_indices;
        for (std::size_t i = 0, j = 0; i < Operand::dim; ++i) {
            if (!this->is_slice_axis[i]) {
                operand_indices[i] = this->indices[j];
                ++j;
            }
        }

        std::array<index_t, Dim> indices;
        const index_t numel = this->numel();
        for (index_t i = 0; i < numel; ++i) {
            util::unravel_index<Dim>(i, this->shape, indices);
            for (std::size_t j = 0; j < Dim; ++j) {
                operand_indices[this->slice_axes[j]] = this->slices[j] * indices[j];
            }
            this->operand[operand_indices] = other[indices];
        }

        return *this;
    }

    NdArraySlice<T, Dim, Operand> &operator=(const T &val) {
        std::array<index_t, Operand::dim> operand_indices;
        for (std::size_t i = 0, j = 0; i < Operand::dim; ++i) {
            if (!this->is_slice_axis[i]) {
                operand_indices[i] = this->indices[j];
                ++j;
            }
        }

        std::array<index_t, Dim> indices;
        const index_t numel = this->numel();
        for (index_t i = 0; i < numel; ++i) {
            util::unravel_index<Dim>(i, this->shape, indices);
            for (std::size_t j = 0; j < Dim; ++j) {
                operand_indices[this->slice_axes[j]] = this->slices[j] * indices[j];
            }
            this->operand[operand_indices] = val;
        }

        return *this;
    }

private:
    Operand &operand;
    const std::array<bool, Operand::dim> is_slice_axis;
    const std::array<index_t, Operand::dim - Dim> indices;
    const std::array<Slice, Dim> slices;
    const std::array<index_t, Dim> slice_axes;
};

std::ostream &operator<<(std::ostream &os, const Slice &slice) {
    os << static_cast<std::string>(slice);
    return os;
}

}  // namespace ndarray

#endif
