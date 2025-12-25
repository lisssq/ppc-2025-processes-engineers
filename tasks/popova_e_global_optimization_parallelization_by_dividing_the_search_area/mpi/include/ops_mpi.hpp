#pragma once
#include "popova_e_global_optimization_parallelization_by_dividing_the_search_area/common/include/common.hpp"
#include "task/include/task.hpp"

namespace popova_e_global_optimization_parallelization_by_dividing_the_search_area {

class PopovaEGlobalOptimizationDividingSearchAreaMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }

  explicit PopovaEGlobalOptimizationDividingSearchAreaMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  // Вычисляет значение функции для оптимизации в точке (x, y)
  double FunctionToOptimize(double x, double y);

  // Вычисляет границы области для текущего процесса
  void ComputeProcessArea(int rank, int size, double &x_start, double &x_end, double &y_start, double &y_end);

  // Грубый поиск с большим шагом
  void CoarseSearch(double x_start, double x_end, double y_start, double y_end, double big_step, double &best_x,
                    double &best_y, double &min_value);

  // Точный поиск в указанной области
  void FineSearch(double x_start, double x_end, double y_start, double y_end, double step, double &best_x,
                  double &best_y, double &min_value);

  // Поиск в остальной области с промежуточным шагом
  void SearchRemainingArea(double x_start, double x_end, double y_start, double y_end, double skip_x_start,
                           double skip_x_end, double skip_y_start, double skip_y_end, double middle_step,
                           double fine_step, double &best_x, double &best_y, double &min_value);

  // Находит глобальный минимум среди всех процессов
  void FindGlobalMinimum(double local_x, double local_y, double local_value, double &global_x, double &global_y,
                         double &global_value);
};

}  // namespace popova_e_global_optimization_parallelization_by_dividing_the_search_area
