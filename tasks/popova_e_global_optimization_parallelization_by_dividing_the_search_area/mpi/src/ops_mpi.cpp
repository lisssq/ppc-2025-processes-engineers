#include "popova_e_global_optimization_parallelization_by_dividing_the_search_area/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <array>
#include <cmath>
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
  return (in.x_max > in.x_min) && (in.y_max > in.y_min) && (in.step > 0);
}

bool PopovaEGlobalOptimizationDividingSearchAreaMPI::PreProcessingImpl() {
  return true;
}

double PopovaEGlobalOptimizationDividingSearchAreaMPI::FunctionToOptimize(double x, double y) {
  const auto &in = GetInput();

  switch (in.func_id) {
    case FunctionType::kParabola1:
      // (x-2)^2 + (y-3)^2
      return ((x - 2.0) * (x - 2.0)) + ((y - 3.0) * (y - 3.0));

    case FunctionType::kParabola2:
      // x^2 + y^2
      return (x * x) + (y * y);

    case FunctionType::kParabola3:
      // (x-1)^2 + (y-1)^2 + 1
      return (((x - 1.0) * (x - 1.0)) + ((y - 1.0) * (y - 1.0))) + 1.0;

    case FunctionType::kParabola4:
      // (x+1)^2 + (y+1)^2
      return ((x + 1.0) * (x + 1.0)) + ((y + 1.0) * (y + 1.0));

    default:
      // По умолчанию используем функцию 1
      return ((x - 2.0) * (x - 2.0)) + ((y - 3.0) * (y - 3.0));
  }
}

void PopovaEGlobalOptimizationDividingSearchAreaMPI::ComputeProcessArea(int rank, int size, double &x_start,
                                                                        double &x_end, double &y_start, double &y_end) {
  const auto &in = GetInput();

  int rows = static_cast<int>(std::sqrt(static_cast<double>(size)));
  while (size % rows != 0 && rows > 0) {
    --rows;
  }
  if (rows == 0) {
    rows = 1;
  }
  int cols = size / rows;

  int row = rank / cols;
  int col = rank % cols;

  double x_range = in.x_max - in.x_min;
  double y_range = in.y_max - in.y_min;
  double x_step = x_range / cols;
  double y_step = y_range / rows;

  x_start = in.x_min + (col * x_step);
  if (col == cols - 1) {
    x_end = in.x_max;
  } else {
    x_end = in.x_min + ((col + 1) * x_step);
  }

  y_start = in.y_min + (row * y_step);
  if (row == rows - 1) {
    y_end = in.y_max;
  } else {
    y_end = in.y_min + ((row + 1) * y_step);
  }
}

void PopovaEGlobalOptimizationDividingSearchAreaMPI::CoarseSearch(double x_start, double x_end, double y_start,
                                                                  double y_end, double big_step, double &best_x,
                                                                  double &best_y, double &min_value) {
  min_value = std::numeric_limits<double>::max();
  best_x = x_start;
  best_y = y_start;

  int x_steps = static_cast<int>((x_end - x_start) / big_step) + 1;
  int y_steps = static_cast<int>((y_end - y_start) / big_step) + 1;

  for (int idx_x = 0; idx_x < x_steps; ++idx_x) {
    double coord_x = x_start + idx_x * big_step;
    if (coord_x > x_end) {
      coord_x = x_end;
    }
    for (int idx_y = 0; idx_y < y_steps; ++idx_y) {
      double coord_y = y_start + idx_y * big_step;
      if (coord_y > y_end) {
        coord_y = y_end;
      }
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
  int x_steps = static_cast<int>((x_end - x_start) / step) + 1;
  int y_steps = static_cast<int>((y_end - y_start) / step) + 1;

  for (int idx_x = 0; idx_x < x_steps; ++idx_x) {
    double coord_x = x_start + idx_x * step;
    if (coord_x > x_end) {
      coord_x = x_end;
    }
    for (int idx_y = 0; idx_y < y_steps; ++idx_y) {
      double coord_y = y_start + idx_y * step;
      if (coord_y > y_end) {
        coord_y = y_end;
      }
      double value = FunctionToOptimize(coord_x, coord_y);
      if (value < min_value) {
        min_value = value;
        best_x = coord_x;
        best_y = coord_y;
      }
    }
  }
}

void PopovaEGlobalOptimizationDividingSearchAreaMPI::SearchRemainingArea(double x_start, double x_end, double y_start,
                                                                         double y_end, double skip_x_start,
                                                                         double skip_x_end, double skip_y_start,
                                                                         double skip_y_end, double middle_step,
                                                                         double fine_step, double &best_x,
                                                                         double &best_y, double &min_value) {
  int x_steps = static_cast<int>((x_end - x_start) / middle_step) + 1;
  int y_steps = static_cast<int>((y_end - y_start) / middle_step) + 1;

  for (int idx_x = 0; idx_x < x_steps; ++idx_x) {
    double coord_x = x_start + idx_x * middle_step;
    if (coord_x > x_end) {
      coord_x = x_end;
    }
    for (int idx_y = 0; idx_y < y_steps; ++idx_y) {
      double coord_y = y_start + idx_y * middle_step;
      if (coord_y > y_end) {
        coord_y = y_end;
      }
      if (coord_x >= skip_x_start && coord_x <= skip_x_end && coord_y >= skip_y_start && coord_y <= skip_y_end) {
        continue;
      }

      double value = FunctionToOptimize(coord_x, coord_y);
      if (value < min_value) {
        double radius = 2.0 * middle_step;
        double local_x_start = std::max(x_start, coord_x - radius);
        double local_x_end = std::min(x_end, coord_x + radius);
        double local_y_start = std::max(y_start, coord_y - radius);
        double local_y_end = std::min(y_end, coord_y + radius);

        FineSearch(local_x_start, local_x_end, local_y_start, local_y_end, fine_step, best_x, best_y, min_value);
      }
    }
  }
}

void PopovaEGlobalOptimizationDividingSearchAreaMPI::FindGlobalMinimum(double local_x, double local_y,
                                                                       double local_value, double &global_x,
                                                                       double &global_y, double &global_value) {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

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

  // шаг 1: грубый поиск с большим шагом
  double big_step = in.step * 5.0;
  double best_x = 0.0;
  double best_y = 0.0;
  double min_value = 0.0;
  CoarseSearch(x_start, x_end, y_start, y_end, big_step, best_x, best_y, min_value);

  // наг 2: точный поиск в окрестности найденной точки
  double search_radius = 3.0 * big_step;
  double refine_x_start = std::max(x_start, best_x - search_radius);
  double refine_x_end = std::min(x_end, best_x + search_radius);
  double refine_y_start = std::max(y_start, best_y - search_radius);
  double refine_y_end = std::min(y_end, best_y + search_radius);

  FineSearch(refine_x_start, refine_x_end, refine_y_start, refine_y_end, in.step, best_x, best_y, min_value);

  // шаг 3: поиск в остальной области для обнаружения других локальных минимумов
  double middle_step = in.step * 3.0;
  SearchRemainingArea(x_start, x_end, y_start, y_end, refine_x_start, refine_x_end, refine_y_start, refine_y_end,
                      middle_step, in.step, best_x, best_y, min_value);

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
