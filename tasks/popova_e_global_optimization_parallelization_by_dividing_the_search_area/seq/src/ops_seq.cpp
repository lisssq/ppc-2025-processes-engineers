#include "popova_e_global_optimization_parallelization_by_dividing_the_search_area/seq/include/ops_seq.hpp"

#include <algorithm>
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

  int x_steps = static_cast<int>((in.x_max - in.x_min) / in.step) + 2;
  int y_steps = static_cast<int>((in.y_max - in.y_min) / in.step) + 2;

  for (int idx_x = 0; idx_x < x_steps; ++idx_x) {
    double coord_x = in.x_min + (static_cast<double>(idx_x) * in.step);
    coord_x = std::min(coord_x, in.x_max);

    for (int idx_y = 0; idx_y < y_steps; ++idx_y) {
      double coord_y = in.y_min + (static_cast<double>(idx_y) * in.step);
      coord_y = std::min(coord_y, in.y_max);

      double value = FunctionToOptimize(coord_x, coord_y);
      if (value < min_value) {
        min_value = value;
        best_x = coord_x;
        best_y = coord_y;
      }
    }
  }
  GetOutput() = std::make_tuple(best_x, best_y, min_value);
  return true;
}

bool PopovaEOptimisationSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace popova_e_global_optimization_parallelization_by_dividing_the_search_area
