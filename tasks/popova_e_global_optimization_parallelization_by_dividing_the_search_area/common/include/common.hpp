#pragma once
#include <cstdint>
#include <string>
#include <tuple>

#include "task/include/task.hpp"

namespace popova_e_global_optimization_parallelization_by_dividing_the_search_area {

enum class FunctionType : std::uint8_t {
  kParabola1 = 1,  // (x-2)^2 + (y-3)^2
  kParabola2 = 2,  // x^2 + y^2,
  kParabola3 = 3,  // (x-1)^2 + (y-1)^2 + 1,
  kParabola4 = 4   // (x+1)^2 + (y+1)^2
};

struct OptimizationInput {
  double x_min{};
  double x_max{};
  double y_min{};
  double y_max{};
  double step{};  // шаг сетки
  FunctionType func_id = FunctionType::kParabola1;
};

using InType = OptimizationInput;
using OutType = std::tuple<double, double, double>;
using TestType = std::tuple<InType, std::string>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace popova_e_global_optimization_parallelization_by_dividing_the_search_area
