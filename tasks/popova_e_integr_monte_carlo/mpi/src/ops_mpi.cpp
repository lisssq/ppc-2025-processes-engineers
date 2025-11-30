#include "popova_e_integr_monte_carlo/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <random>

#include "popova_e_integr_monte_carlo/common/include/common.hpp"

namespace popova_e_integr_monte_carlo {

PopovaEIntegrMonteCarloMPI::PopovaEIntegrMonteCarloMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool PopovaEIntegrMonteCarloMPI::ValidationImpl() {
  const auto &[a, b, n] = GetInput();
  return (a < b) && (n > 0);
}

bool PopovaEIntegrMonteCarloMPI::PreProcessingImpl() {
  const auto &[a, b, n] = GetInput();
  a_ = a;
  b_ = b;
  point_count_ = n;

  return true;
}

bool PopovaEIntegrMonteCarloMPI::RunImpl() {
  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int local_point_count = point_count_ / size;
  int extra_points = point_count_ % size;

  if (rank < extra_points) {
    local_point_count++;
  }

  // псевдослучайная последовательность
  const double magic_constant = 0.75487766624669276;
  double current = 0.1 + 0.8 * rank / size;

  double local_sum = 0.0;
  for (int i = 0; i < local_point_count; ++i) {
    current = std::fmod(current + magic_constant, 1.0);
    double x = a_ + (b_ - a_) * current;

    double fx = (x * x * x) - (4 * x);
    local_sum += fx;
  }

  double total_sum = 0.0;
  MPI_Reduce(&local_sum, &total_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

  double integral = 0.0;
  if (rank == 0) {
    double sredn = total_sum / static_cast<double>(point_count_);
    integral = (b_ - a_) * sredn;
  }

  MPI_Bcast(&integral, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  GetOutput() = integral;
  return true;
}

bool PopovaEIntegrMonteCarloMPI::PostProcessingImpl() {
  return true;
}

}  // namespace popova_e_integr_monte_carlo
