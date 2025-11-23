#include "popova_e_integr_monte_carlo/seq/include/ops_seq.hpp"

#include "popova_e_integr_monte_carlo/common/include/common.hpp"
#include "util/include/util.hpp"

namespace popova_e_integr_monte_carlo {

PopovaEIntegrMonteCarloSEQ::PopovaEIntegrMonteCarloSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool PopovaEIntegrMonteCarloSEQ::ValidationImpl() {
  // const auto &[a, b, n] = GetInput();
  // a_ = a;
  // b_ = b;
  // point_count = n;

  // return (a_ < b_) && (point_count > 0);

  const auto &[a, b, n] = GetInput();
  return (a < b) && (n > 0);
}

bool PopovaEIntegrMonteCarloSEQ::PreProcessingImpl() {
  const auto &[a, b, n] = GetInput();
  a_ = a;
  b_ = b;
  point_count = n;

  return true;
}

bool PopovaEIntegrMonteCarloSEQ::RunImpl() {
  std::mt19937 generate(12345);
  std::uniform_real_distribution<double> dist(a_, b_);

  double sum = 0.0;
  for (int i = 0; i < point_count; ++i) {
    double x = dist(generate);
    // интеграл f(x) = x^3 - 4x
    double fx = x * x * x - 4 * x;
    sum += fx;
  }
  double sredn = sum / static_cast<double>(point_count);
  double integral = (b_ - a_) * sredn;

  GetOutput() = integral;

  return true;
}

bool PopovaEIntegrMonteCarloSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace popova_e_integr_monte_carlo
