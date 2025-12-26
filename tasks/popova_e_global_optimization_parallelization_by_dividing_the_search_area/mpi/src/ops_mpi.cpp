#include "popova_e_global_optimization_parallelization_by_dividing_the_search_area/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <array>
#include <limits>
#include <tuple>

#include "popova_e_global_optimization_parallelization_by_dividing_the_search_area/common/include/common.hpp"

namespace popova_e_global_optimization_parallelization_by_dividing_the_search_area {

PopovaEGlobalOptimizationDividingSearchAreaMPI::PopovaEGlobalOptimizationDividingSearchAreaMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = std::make_tuple(0.0, 0.0, std::numeric_limits<double>::max());
}

bool PopovaEGlobalOptimizationDividingSearchAreaMPI::ValidationImpl() {
  const auto &in = GetInput();
  return ((in.x_max > in.x_min) && (in.y_max > in.y_min) && (in.step > 0));
}

bool PopovaEGlobalOptimizationDividingSearchAreaMPI::PreProcessingImpl() {
  return true;
}

double PopovaEGlobalOptimizationDividingSearchAreaMPI::FunctionToOptimize(double x, double y) {
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

void PopovaEGlobalOptimizationDividingSearchAreaMPI::ComputeProcessArea(int rank, int size, double &x_start,
                                                                        double &x_end, double &y_start, double &y_end) {
  const auto &in = GetInput();

  double x_range = in.x_max - in.x_min;
  double x_step = x_range / size;

  x_start = in.x_min + (rank * x_step);
  if (rank == size - 1) {
    x_end = in.x_max;
  } else {
    x_end = in.x_min + ((rank + 1) * x_step);
  }

  y_start = in.y_min;
  y_end = in.y_max;
}

void PopovaEGlobalOptimizationDividingSearchAreaMPI::CoarseSearch(double x_start, double x_end, double y_start,
                                                                  double y_end, double big_step, double &best_x,
                                                                  double &best_y, double &min_value) {
  min_value = std::numeric_limits<double>::max();
  best_x = x_start;
  best_y = y_start;

  int x_steps = static_cast<int>((x_end - x_start) / big_step) + 2;
  int y_steps = static_cast<int>((y_end - y_start) / big_step) + 2;

  for (int idx_x = 0; idx_x < x_steps; ++idx_x) {
    double coord_x = x_start + (static_cast<double>(idx_x) * big_step);
    coord_x = std::min(coord_x, x_end);

    for (int idx_y = 0; idx_y < y_steps; ++idx_y) {
      double coord_y = y_start + (static_cast<double>(idx_y) * big_step);
      coord_y = std::min(coord_y, y_end);

      double value = FunctionToOptimize(coord_x, coord_y);
      if (value < min_value) {
        min_value = value;
        best_x = coord_x;
        best_y = coord_y;
      }
    }
  }
}

void PopovaEGlobalOptimizationDividingSearchAreaMPI::FineSearch(double x_start, double x_end, double y_start,
                                                                double y_end, double step, double &best_x,
                                                                double &best_y, double &min_value) {
  int x_steps = static_cast<int>((x_end - x_start) / step) + 2;
  int y_steps = static_cast<int>((y_end - y_start) / step) + 2;

  for (int idx_x = 0; idx_x < x_steps; ++idx_x) {
    double coord_x = x_start + static_cast<double>(idx_x) * step;
    coord_x = std::min(coord_x, x_end);

    for (int idx_y = 0; idx_y < y_steps; ++idx_y) {
      double coord_y = y_start + static_cast<double>(idx_y) * step;
      coord_y = std::min(coord_y, y_end);

      double value = FunctionToOptimize(coord_x, coord_y);
      if (value < min_value) {
        min_value = value;
        best_x = coord_x;
        best_y = coord_y;
      }
    }
  }
}

void PopovaEGlobalOptimizationDividingSearchAreaMPI::FindGlobalMinimum(double local_x, double local_y,
                                                                       double local_value, double &global_x,
                                                                       double &global_y, double &global_value) {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  int size = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  struct {
    double value{};
    int rank{};
  } local_min{};
  struct {
    double value{};
    int rank{};
  } global_min{};

  local_min.value = local_value;
  local_min.rank = rank;

  MPI_Allreduce(&local_min, &global_min, 1, MPI_DOUBLE_INT, MPI_MINLOC, MPI_COMM_WORLD);

  std::array<double, 2> coords = {local_x, local_y};
  MPI_Bcast(coords.data(), 2, MPI_DOUBLE, global_min.rank, MPI_COMM_WORLD);

  global_x = coords[0];
  global_y = coords[1];
  global_value = global_min.value;
}

bool PopovaEGlobalOptimizationDividingSearchAreaMPI::RunImpl() {
  const auto &in = GetInput();

  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  int size = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  double x_start = 0.0;
  double x_end = 0.0;
  double y_start = 0.0;
  double y_end = 0.0;
  ComputeProcessArea(rank, size, x_start, x_end, y_start, y_end);

  double big_step = in.step * 5.0;
  double best_x = x_start;
  double best_y = y_start;
  double min_value = std::numeric_limits<double>::max();

  CoarseSearch(x_start, x_end, y_start, y_end, big_step, best_x, best_y, min_value);
  FineSearch(x_start, x_end, y_start, y_end, in.step, best_x, best_y, min_value);

  double global_x = 0.0;
  double global_y = 0.0;
  double global_value = 0.0;
  FindGlobalMinimum(best_x, best_y, min_value, global_x, global_y, global_value);

  GetOutput() = std::make_tuple(global_x, global_y, global_value);
  return true;
}

bool PopovaEGlobalOptimizationDividingSearchAreaMPI::PostProcessingImpl() {
  return true;
}

}  // namespace popova_e_global_optimization_parallelization_by_dividing_the_search_area
