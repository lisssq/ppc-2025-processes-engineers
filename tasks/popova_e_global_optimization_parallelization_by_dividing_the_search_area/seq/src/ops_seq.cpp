#include "popova_e_global_optimization_parallelization_by_dividing_the_search_area/seq/include/ops_seq.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <tuple>

#include "popova_e_global_optimization_parallelization_by_dividing_the_search_area/common/include/common.hpp"

namespace popova_e_global_optimization_parallelization_by_dividing_the_search_area {

PopovaEOptimisationSEQ::PopovaEOptimisationSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = std::make_tuple(0.0, 0.0, std::numeric_limits<double>::max());
}

bool PopovaEOptimisationSEQ::ValidationImpl() {
  const auto &in = GetInput();
  return (in.x_max > in.x_min) && (in.y_max > in.y_min) && (in.step > 0);
}

bool PopovaEOptimisationSEQ::PreProcessingImpl() {
  return true;
}

double PopovaEOptimisationSEQ::FunctionToOptimize(double x, double y) {
  const auto &in = GetInput();

  switch (in.func_id) {
    case FunctionType::kParabola1:
      return ((x - 2.0) * (x - 2.0)) + ((y - 3.0) * (y - 3.0));
    case FunctionType::kParabola2:
      return (x * x) + (y * y);
    case FunctionType::kParabola3:
      return (((x - 1.0) * (x - 1.0)) + ((y - 1.0) * (y - 1.0))) + 1.0;
    case FunctionType::kParabola4:
      return ((x + 1.0) * (x + 1.0)) + ((y + 1.0) * (y + 1.0));
    default:
      return ((x - 2.0) * (x - 2.0)) + ((y - 3.0) * (y - 3.0));
  }
}

bool PopovaEOptimisationSEQ::RunImpl() {
  const auto &in = GetInput();

  double best_x = in.x_min;
  double best_y = in.y_min;
  double min_value = std::numeric_limits<double>::max();

  const double epsilon = 1e-12;

  // Простой и надежный алгоритм
  double x = in.x_min;
  while (x <= in.x_max + epsilon) {
    double current_x = std::min(x, in.x_max);

    double y = in.y_min;
    while (y <= in.y_max + epsilon) {
      double current_y = std::min(y, in.y_max);

      double value = FunctionToOptimize(current_x, current_y);
      if (value < min_value) {
        min_value = value;
        best_x = current_x;
        best_y = current_y;
      }

      y += in.step;
    }

    x += in.step;
  }

  GetOutput() = std::make_tuple(best_x, best_y, min_value);
  return true;
}

bool PopovaEOptimisationSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace popova_e_global_optimization_parallelization_by_dividing_the_search_area
