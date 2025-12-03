#pragma once

#include <string>
#include <tuple>

#include "task/include/task.hpp"

namespace popova_e_integr_monte_carlo {

using InType = std::tuple<double, double, int, int>;
using OutType = double;
using TestType = std::tuple<std::tuple<double, double, int, int>, std::string>;
using BaseTask = ppc::task::Task<InType, OutType>;

enum FuncType {
  linear_func = 0,     // 2x + 7
  quadratic_func = 1,  // 5x - 3x^2 + 7
  cubic_func = 2,      // x^3 - 4x
  cos_func = 3,        // cos(2x)
  exp_func = 4         // 2x * exp(-2x) + 4
};

}  // namespace popova_e_integr_monte_carlo
