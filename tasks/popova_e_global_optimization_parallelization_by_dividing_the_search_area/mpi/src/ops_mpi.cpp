#include "popova_e_global_optimization_parallelization_by_dividing_the_search_area/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <limits>

#include "popova_e_global_optimization_parallelization_by_dividing_the_search_area/common/include/common.hpp"

namespace popova_e_global_optimization_parallelization_by_dividing_the_search_area {

PopovaEOptimisationFieldMPI::PopovaEOptimisationFieldMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = std::make_tuple(0.0, 0.0, std::numeric_limits<double>::max());
}

bool PopovaEOptimisationFieldMPI::ValidationImpl() {
  const auto &in = GetInput();
  return (in.x_max > in.x_min) && (in.y_max > in.y_min) && (in.step > 0);
}

bool PopovaEOptimisationFieldMPI::PreProcessingImpl() {
  return true;
}

double PopovaEOptimisationFieldMPI::FunctionToOptimize(double x, double y) {
  return (x - 2.0) * (x - 2.0) + (y - 3.0) * (y - 3.0);
}

bool PopovaEOptimisationFieldMPI::RunImpl() {
  const auto &in = GetInput();

  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  auto compute_range = [&](double x_min, double x_max, double &x_start, double &x_end) {
    const double x_range = x_max - x_min;
    x_start = x_min + rank * (x_range / size);
    if (rank == size - 1) {
      x_end = x_max;
    } else {
      x_end = x_min + (rank + 1) * (x_range / size);
    }
    if (x_start > x_end) {
      x_start = x_min;
      x_end = x_min;
    }
  };

  auto parallel_search = [&](double x_min, double x_max, double y_min, double y_max, double step, double &f_min_local,
                             double &x_best_local, double &y_best_local) {
    double x_start, x_end;
    compute_range(x_min, x_max, x_start, x_end);
    f_min_local = std::numeric_limits<double>::max();
    x_best_local = x_start;
    y_best_local = y_min;
    for (double x = x_start; x <= x_end; x += step) {
      for (double y = y_min; y <= y_max; y += step) {
        double f = FunctionToOptimize(x, y);
        if (f < f_min_local) {
          f_min_local = f;
          x_best_local = x;
          y_best_local = y;
        }
      }
    }
  };

  auto reduce_global_min = [&](double f_min_local, double x_best_local, double y_best_local, double global[3]) {
    double local[3] = {f_min_local, x_best_local, y_best_local};

    double *all = nullptr;
    if (rank == 0) {
      all = new double[3 * size];
    }

    MPI_Gather(local, 3, MPI_DOUBLE, all, 3, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0) {
      global[0] = all[0];
      global[1] = all[1];
      global[2] = all[2];
      for (int i = 1; i < size; ++i) {
        if (all[i * 3] < global[0]) {
          global[0] = all[i * 3];
          global[1] = all[i * 3 + 1];
          global[2] = all[i * 3 + 2];
        }
      }
      delete[] all;
    }

    MPI_Bcast(global, 3, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  };

  double f_min_local = 0.0, x_best_local = 0.0, y_best_local = 0.0;
  parallel_search(in.x_min, in.x_max, in.y_min, in.y_max, in.step, f_min_local, x_best_local, y_best_local);

  double global_coarse[3] = {0.0, 0.0, 0.0};
  reduce_global_min(f_min_local, x_best_local, y_best_local, global_coarse);

  const double refine_step = std::max(in.step / 2.0, std::numeric_limits<double>::epsilon());
  const double x_ref_min = std::max(in.x_min, global_coarse[1] - in.step);
  const double x_ref_max = std::min(in.x_max, global_coarse[1] + in.step);
  const double y_ref_min = std::max(in.y_min, global_coarse[2] - in.step);
  const double y_ref_max = std::min(in.y_max, global_coarse[2] + in.step);

  parallel_search(x_ref_min, x_ref_max, y_ref_min, y_ref_max, refine_step, f_min_local, x_best_local, y_best_local);

  double global_refined[3] = {0.0, 0.0, 0.0};
  reduce_global_min(f_min_local, x_best_local, y_best_local, global_refined);

  GetOutput() = std::make_tuple(global_refined[1], global_refined[2], global_refined[0]);

  MPI_Barrier(MPI_COMM_WORLD);
  return true;
}

bool PopovaEOptimisationFieldMPI::PostProcessingImpl() {
  return true;
}

}  // namespace popova_e_global_optimization_parallelization_by_dividing_the_search_area
