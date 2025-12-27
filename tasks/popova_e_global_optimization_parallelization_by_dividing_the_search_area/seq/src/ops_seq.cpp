#include "popova_e_global_optimization_parallelization_by_dividing_the_search_area/seq/include/ops_seq.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
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

double PopovaEOptimisationSEQ::FunctionToOptimize(double px, double py) {
  return ((px - 2.0) * (px - 2.0)) + ((py - 3.0) * (py - 3.0));
}

bool PopovaEOptimisationSEQ::RunImpl() {
  const auto &in = GetInput();
  const double safe_step = std::max(in.step, std::numeric_limits<double>::epsilon());

  double f_min = std::numeric_limits<double>::max();
  double x_best = in.x_min;
  double y_best = in.y_min;

  const auto num_x = static_cast<std::size_t>(std::floor(((in.x_max - in.x_min) / safe_step) + 0.5)) + 1U;
  const auto num_y = static_cast<std::size_t>(std::floor(((in.y_max - in.y_min) / safe_step) + 0.5)) + 1U;

  for (std::size_t ix = 0; ix < num_x; ++ix) {
    double px = in.x_min + (static_cast<double>(ix) * safe_step);
    px = std::min(px, in.x_max);
    for (std::size_t iy = 0; iy < num_y; ++iy) {
      double py = in.y_min + (static_cast<double>(iy) * safe_step);
      py = std::min(py, in.y_max);
      double fval = FunctionToOptimize(px, py);
      if (fval < f_min) {
        f_min = fval;
        x_best = px;
        y_best = py;
      }
    }
  }

  const double refine_step = std::max(in.step / 2.0, std::numeric_limits<double>::epsilon());
  const double x_ref_min = std::max(in.x_min, x_best - in.step);
  const double x_ref_max = std::min(in.x_max, x_best + in.step);
  const double y_ref_min = std::max(in.y_min, y_best - in.step);
  const double y_ref_max = std::min(in.y_max, y_best + in.step);

  const auto ref_num_x = static_cast<std::size_t>(std::floor(((x_ref_max - x_ref_min) / refine_step) + 0.5)) + 1U;
  const auto ref_num_y = static_cast<std::size_t>(std::floor(((y_ref_max - y_ref_min) / refine_step) + 0.5)) + 1U;

  for (std::size_t ix = 0; ix < ref_num_x; ++ix) {
    double px = x_ref_min + (static_cast<double>(ix) * refine_step);
    px = std::min(px, x_ref_max);
    for (std::size_t iy = 0; iy < ref_num_y; ++iy) {
      double py = y_ref_min + (static_cast<double>(iy) * refine_step);
      py = std::min(py, y_ref_max);
      double fval = FunctionToOptimize(px, py);
      if (fval < f_min) {
        f_min = fval;
        x_best = px;
        y_best = py;
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
