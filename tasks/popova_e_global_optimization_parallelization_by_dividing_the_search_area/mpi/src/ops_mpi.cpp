#include "popova_e_global_optimization_parallelization_by_dividing_the_search_area/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <tuple>
#include <vector>

#include "popova_e_global_optimization_parallelization_by_dividing_the_search_area/common/include/common.hpp"

namespace popova_e_global_optimization_parallelization_by_dividing_the_search_area {

namespace {

struct XRange {
  double start{};
  double end{};
  bool has_work{};
};

struct LocalBest {
  double f{std::numeric_limits<double>::max()};
  double x{};
  double y{};
};

double SafeStep(double step) {
  return std::max(step, std::numeric_limits<double>::epsilon());
}

double FunctionToOptimizeImpl(double px, double py) {
  return ((px - 2.0) * (px - 2.0)) + ((py - 3.0) * (py - 3.0));
}

XRange ComputeXRange(int rank, int world_size, double x_min, double x_max, double step) {
  const double safe_step = SafeStep(step);
  const double total_x_range = x_max - x_min;

  if (total_x_range < safe_step / 2.0) {
    if (rank == 0) {
      return XRange{x_min, x_max, true};
    }
    return XRange{x_min, x_min, false};
  }

  const std::size_t total_points = static_cast<std::size_t>(std::floor((total_x_range / safe_step) + 0.5)) + 1U;
  if (total_points == 0U) {
    return XRange{x_min, x_min, false};
  }

  const std::size_t world = static_cast<std::size_t>(world_size);
  const std::size_t r = static_cast<std::size_t>(rank);
  const std::size_t base_points = total_points / world;
  const std::size_t remainder = total_points % world;

  std::size_t my_points = base_points;
  if (r < remainder) {
    my_points += 1U;
  }

  std::size_t prefix_points = r * base_points;
  if (r < remainder) {
    prefix_points += r;
  } else {
    prefix_points += remainder;
  }

  if (my_points == 0U) {
    return XRange{x_min, x_min, false};
  }

  const double x_start = x_min + (static_cast<double>(prefix_points) * safe_step);
  double x_end = x_min + (static_cast<double>(prefix_points + my_points - 1U) * safe_step);
  x_end = std::min(x_end, x_max);

  return XRange{x_start, x_end, true};
}

LocalBest ParallelSearch(int rank, int world_size, double x_min, double x_max, double y_min, double y_max,
                         double step) {
  const double safe_step = SafeStep(step);

  const XRange xr = ComputeXRange(rank, world_size, x_min, x_max, step);
  if (!xr.has_work) {
    return LocalBest{std::numeric_limits<double>::max(), x_min, y_min};
  }

  LocalBest best{std::numeric_limits<double>::max(), x_min, y_min};

  const auto num_x = static_cast<std::size_t>(std::floor(((xr.end - xr.start) / safe_step) + 1.5));
  const auto num_y = static_cast<std::size_t>(std::floor(((y_max - y_min) / safe_step) + 1.5));

  for (std::size_t ix = 0; ix < num_x; ++ix) {
    double px = xr.start + (static_cast<double>(ix) * safe_step);
    px = std::min(px, xr.end);
    for (std::size_t iy = 0; iy < num_y; ++iy) {
      double py = y_min + (static_cast<double>(iy) * safe_step);
      py = std::min(py, y_max);
      const double fval = FunctionToOptimizeImpl(px, py);
      if (fval < best.f) {
        best.f = fval;
        best.x = px;
        best.y = py;
      }
    }
  }

  return best;
}

std::array<double, 3> ReduceGlobalMin(int rank, int world_size, const LocalBest &local_best) {
  std::array<double, 3> local_data = {local_best.f, local_best.x, local_best.y};

  std::vector<double> all_data;
  if (rank == 0) {
    all_data.resize(static_cast<std::size_t>(world_size) * 3U);
  }

  MPI_Gather(local_data.data(), 3, MPI_DOUBLE, all_data.data(), 3, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  std::array<double, 3> global_out = local_data;
  if (rank == 0) {
    global_out[0] = all_data[0];
    global_out[1] = all_data[1];
    global_out[2] = all_data[2];

    for (int pi = 1; pi < world_size; ++pi) {
      const auto idx = static_cast<std::size_t>(pi) * 3U;
      if (all_data[idx] < global_out[0]) {
        global_out[0] = all_data[idx];
        global_out[1] = all_data[idx + 1U];
        global_out[2] = all_data[idx + 2U];
      }
    }
  }

  MPI_Bcast(global_out.data(), 3, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  return global_out;
}

}  // namespace

PopovaEGlobalOptimizationMPI::PopovaEGlobalOptimizationMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = std::make_tuple(0.0, 0.0, std::numeric_limits<double>::max());
}

bool PopovaEGlobalOptimizationMPI::ValidationImpl() {
  const auto &in = GetInput();
  return (in.x_max > in.x_min) && (in.y_max > in.y_min) && (in.step > 0);
}

bool PopovaEGlobalOptimizationMPI::PreProcessingImpl() {
  return true;
}

double PopovaEGlobalOptimizationMPI::FunctionToOptimize(double px, double py) {
  return FunctionToOptimizeImpl(px, py);
}

bool PopovaEGlobalOptimizationMPI::RunImpl() {
  const auto &in = GetInput();

  int rank = 0;
  int world_size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  const LocalBest local_coarse = ParallelSearch(rank, world_size, in.x_min, in.x_max, in.y_min, in.y_max, in.step);
  const std::array<double, 3> global_coarse = ReduceGlobalMin(rank, world_size, local_coarse);

  const double refine_step = std::max(in.step / 2.0, std::numeric_limits<double>::epsilon());
  const double x_ref_min = std::max(in.x_min, global_coarse[1] - in.step);
  const double x_ref_max = std::min(in.x_max, global_coarse[1] + in.step);
  const double y_ref_min = std::max(in.y_min, global_coarse[2] - in.step);
  const double y_ref_max = std::min(in.y_max, global_coarse[2] + in.step);

  const LocalBest local_refined =
      ParallelSearch(rank, world_size, x_ref_min, x_ref_max, y_ref_min, y_ref_max, refine_step);
  const std::array<double, 3> global_refined = ReduceGlobalMin(rank, world_size, local_refined);

  GetOutput() = std::make_tuple(global_refined[1], global_refined[2], global_refined[0]);

  return true;
}

bool PopovaEGlobalOptimizationMPI::PostProcessingImpl() {
  return true;
}

}  // namespace popova_e_global_optimization_parallelization_by_dividing_the_search_area
