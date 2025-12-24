#pragma once
#include <utility>
#include <vector>

#include "popova_e_vertical_ribbon_scheme_matrix_multiplication_by_vector/common/include/common.hpp"
#include "task/include/task.hpp"

namespace popova_e_vertical_ribbon_scheme_matrix_multiplication_by_vector {

class PopovaEVerticalRibbonSchemeMatrixMultiplicationByVectorMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit PopovaEVerticalRibbonSchemeMatrixMultiplicationByVectorMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  int rows_ = 0;
  int cols_ = 0;
};

}  // namespace popova_e_vertical_ribbon_scheme_matrix_multiplication_by_vector
