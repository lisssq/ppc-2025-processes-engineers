#pragma once
#include "popova_e_global_optimization_parallelization_by_dividing_the_search_area/common/include/common.hpp"
#include "task/include/task.hpp"

namespace popova_e_global_optimization_parallelization_by_dividing_the_search_area {

class PopovaEOptimisationFieldMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }

  explicit PopovaEOptimisationFieldMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  double FunctionToOptimize(double x, double y);
};

}  // namespace popova_e_global_optimization_parallelization_by_dividing_the_search_area
