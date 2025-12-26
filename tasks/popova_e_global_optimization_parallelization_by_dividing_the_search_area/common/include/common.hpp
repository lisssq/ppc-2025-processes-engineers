#pragma once
#include <string>
#include <tuple>

#include "task/include/task.hpp"

namespace popova_e_global_optimization_parallelization_by_dividing_the_search_area {

struct OptimizationInput {
  double x_min;
  double x_max;
  double y_min;
  double y_max;
  double step;
};

using InType = OptimizationInput;
using OutType = std::tuple<double, double, double>;
using TestType = std::tuple<InType, std::string>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace popova_e_global_optimization_parallelization_by_dividing_the_search_area
