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
  // std::mt19937 generate(12345);  // NOLINT(cert-msc51-cpp)
  std::random_device rd;
  std::mt19937 generate(rd());
  std::uniform_real_distribution<double> dist(a_, b_);

  double sum = 0.0;
  for (int i = 0; i < point_count_; ++i) {
    double x = dist(generate);

    //  // Детерминированная генерация точек
    // double t = static_cast<double>(i) / point_count_;
    // double x = a_ + (b_ - a_) * t;

    // интеграл f(x) = x^3 - 4x
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
