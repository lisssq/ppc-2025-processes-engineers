#pragma once

#include "popova_e_integr_monte_carlo/common/include/common.hpp"
#include "task/include/task.hpp"

#include <random>


namespace popova_e_integr_monte_carlo {

class PopovaEIntegrMonteCarloSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit PopovaEIntegrMonteCarloSEQ(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;


  int point_count{};
  double a_{};
  double b_{};

};

}  // namespace popova_e_integr_monte_carlo
