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
  const auto &[a, b, n, func_id] = GetInput();
  return (a < b) && (n > 0) && (func_id >= 0) && (func_id <= 4);
}

bool PopovaEIntegrMonteCarloSEQ::PreProcessingImpl() {
  const auto &[a, b, n, func_id] = GetInput();
  a_ = a;
  b_ = b;
  point_count_ = n;
  func_id_ = func_id;

  return true;
}

bool PopovaEIntegrMonteCarloSEQ::RunImpl() {
  const double magic_constant = 0.75487766624669276;
  double current = 0.5;

  double sum = 0.0;
  for (int i = 0; i < point_count_; ++i) {
    // current = std::fmod(current + magic_constant, 1.0);
    current += magic_constant;
    if (current >= 1.0) {
      current -= 1.0;  // достаточно отнять 1, потому что magic_constant < 1
    }

    double x = a_ + (b_ - a_) * current;

    double fx = 0.0;
    fx = FunctionPair::function(func_id_, x);

    // switch (func_id_) {
    //   case linear_func:
    //     fx = (2 * x) + 7;
    //     break;
    //   case quadratic_func:
    //     fx = (5 * x) - (3 * x * x) + 7;
    //     break;
    //   case cubic_func:
    //     fx = (x * x * x) - (4 * x);
    //     break;
    //   case cos_func:
    //     fx = std::cos(2 * x);
    //     break;
    //   case exp_func:
    //     fx = (2 * x) * (std::exp(-2 * x)) + 4;
    //     break;
    // }

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
