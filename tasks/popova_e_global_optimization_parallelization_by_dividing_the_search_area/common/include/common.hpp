#pragma once
#include <string>
#include <tuple>

#include "task/include/task.hpp"

namespace popova_e_global_optimization_parallelization_by_dividing_the_search_area {

// Типы функций для оптимизации
enum class FunctionType {
  kParabola1 = 1,  // (x-2)² + (y-3)², минимум в (2, 3), значение 0
  kParabola2 = 2,  // x² + y², минимум в (0, 0), значение 0
  kParabola3 = 3,  // (x-1)² + (y-1)² + 1, минимум в (1, 1), значение 1
  kParabola4 = 4   // (x+1)² + (y+1)², минимум в (-1, -1), значение 0
};

struct OptimizationInput {
  double x_min;
  double x_max;
  double y_min;
  double y_max;
  double step;                                      // шаг сетки
  FunctionType func_id = FunctionType::kParabola1;  // идентификатор функции для оптимизации
};

using InType = OptimizationInput;
using OutType = std::tuple<double, double, double>;
using TestType = std::tuple<InType, std::string>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace popova_e_global_optimization_parallelization_by_dividing_the_search_area
