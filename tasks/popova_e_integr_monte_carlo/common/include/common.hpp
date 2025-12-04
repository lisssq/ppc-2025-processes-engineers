#pragma once

#include <string>
#include <tuple>

#include "task/include/task.hpp"

namespace popova_e_integr_monte_carlo {

using InType = std::tuple<double, double, int, int>;
using OutType = double;
using TestType = std::tuple<std::tuple<double, double, int, int>, std::string>;
using BaseTask = ppc::task::Task<InType, OutType>;

enum FuncType {
  linear_func = 0,     // 2x + 7
  quadratic_func = 1,  // 5x - 3x^2 + 7
  cubic_func = 2,      // x^3 - 4x
  cos_func = 3,        // cos(2x)
  exp_func = 4         // 2x * exp(-2x) + 4
};

class FunctionPair {
 public:
  static double function(int func_id, double x) {
    switch (func_id) {
      case linear_func:
        return 2.0 * x + 7.0;
      case quadratic_func:
        return 5.0 * x - 3.0 * x * x + 7.0;
      case cubic_func:
        return x * x * x - 4.0 * x;
      case cos_func:
        return std::cos(2.0 * x);
      case exp_func:
        return 2.0 * x * std::exp(-2.0 * x) + 4.0;
      default:
        return 0.0;
    }
  }

  static double integral(int func_id, double x) {
    switch (func_id) {
      case linear_func:
        return x * x + 7.0 * x;  
      case quadratic_func:
        return 2.5 * x * x - x * x * x + 7.0 * x;  
      case cubic_func:
        return 0.25 * x * x * x * x - 2.0 * x * x;  
      case cos_func:
        return 0.5 * std::sin(2.0 * x); 
      case exp_func:
        return -(x + 0.5) * std::exp(-2.0 * x) + 4.0 * x;  
      default:
        return 0.0;
    }
  }
};

}  // namespace popova_e_integr_monte_carlo
