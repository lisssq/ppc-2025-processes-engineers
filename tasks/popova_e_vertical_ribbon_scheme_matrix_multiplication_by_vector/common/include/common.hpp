#pragma once

#include <string>
#include <tuple>

#include "task/include/task.hpp"

namespace popova_e_vertical_ribbon_scheme_matrix_multiplication_by_vector {

using InType = std::pair<int, int>;  // rows, cols
using OutType = std::vector<double>;
using TestType = std::tuple<std::pair<int, int>, std::string>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace popova_e_vertical_ribbon_scheme_matrix_multiplication_by_vector
