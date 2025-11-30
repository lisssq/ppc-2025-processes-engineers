#include "popova_e_integr_monte_carlo/seq/include/ops_seq.hpp"

#include <cmath>
#include <random>

#include "popova_e_integr_monte_carlo/common/include/common.hpp"

namespace popova_e_integr_monte_carlo {

PopovaEIntegrMonteCarloSEQ::PopovaEIntegrMonteCarloSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool PopovaEIntegrMonteCarloSEQ::ValidationImpl() {
  const auto &[a, b, n] = GetInput();
  return (a < b) && (n > 0);
}

bool PopovaEIntegrMonteCarloSEQ::PreProcessingImpl() {
  const auto &[a, b, n] = GetInput();
  a_ = a;
  b_ = b;
  point_count_ = n;

  return true;
}

bool PopovaEIntegrMonteCarloSEQ::RunImpl() {
  const double magic_constant = 0.75487766624669276;
  double current = 0.5;

  double sum = 0.0;
  for (int i = 0; i < point_count_; ++i) {
    current = std::fmod(current + magic_constant, 1.0);
    double x = a_ + (b_ - a_) * current;

    double fx = (x * x * x) - (4 * x);
    sum += fx;
  }

  double sredn = sum / static_cast<double>(point_count_);
  double integral = (b_ - a_) * sredn;
  GetOutput() = integral;

  return true;
}

bool PopovaEIntegrMonteCarloSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace popova_e_integr_monte_carlo
