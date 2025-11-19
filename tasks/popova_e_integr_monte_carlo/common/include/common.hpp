#pragma once

#include <string>
#include <tuple>

#include "task/include/task.hpp"

namespace popova_e_integr_monte_carlo {

using InType = std::tuple<double, double, int>;
using OutType = double;
using TestType = std::tuple<std::tuple<double, double, int>, std::string>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace popova_e_integr_monte_carlo
