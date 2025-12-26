#include "popova_e_global_optimization_parallelization_by_dividing_the_search_area/seq/include/ops_seq.hpp"

#include <algorithm>
#include <limits>

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
  return (x - 2.0) * (x - 2.0) + (y - 3.0) * (y - 3.0);
}

bool PopovaEOptimisationSEQ::RunImpl() {
  const auto &in = GetInput();
  double f_min = std::numeric_limits<double>::max();
  double x_best = in.x_min;
  double y_best = in.y_min;

  for (double x = in.x_min; x <= in.x_max; x += in.step) {
    for (double y = in.y_min; y <= in.y_max; y += in.step) {
      double f = FunctionToOptimize(x, y);
      if (f < f_min) {
        f_min = f;
        x_best = x;
        y_best = y;
      }
    }
  }

  const double refine_step = std::max(in.step / 2.0, std::numeric_limits<double>::epsilon());
  const double x_ref_min = std::max(in.x_min, x_best - in.step);
  const double x_ref_max = std::min(in.x_max, x_best + in.step);
  const double y_ref_min = std::max(in.y_min, y_best - in.step);
  const double y_ref_max = std::min(in.y_max, y_best + in.step);

  for (double x = x_ref_min; x <= x_ref_max; x += refine_step) {
    for (double y = y_ref_min; y <= y_ref_max; y += refine_step) {
      double f = FunctionToOptimize(x, y);
      if (f < f_min) {
        f_min = f;
        x_best = x;
        y_best = y;
      }
    }
  }

  GetOutput() = std::make_tuple(x_best, y_best, f_min);
  return true;
}

bool PopovaEOptimisationSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace popova_e_global_optimization_parallelization_by_dividing_the_search_area
