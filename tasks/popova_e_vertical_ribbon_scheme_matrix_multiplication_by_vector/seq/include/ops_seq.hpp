#pragma once

#include "popova_e_vertical_ribbon_scheme_matrix_multiplication_by_vector/common/include/common.hpp"
#include "task/include/task.hpp"

namespace popova_e_vertical_ribbon_scheme_matrix_multiplication_by_vector {

class PopovaEVerticalRibbonSchemeMatrixMultiplicationByVectorSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit PopovaEVerticalRibbonSchemeMatrixMultiplicationByVectorSEQ(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;


  std::vector<std::vector<double>> matrix_;
  std::vector<double> vector_;
};

}  // namespace popova_e_vertical_ribbon_scheme_matrix_multiplication_by_vector
