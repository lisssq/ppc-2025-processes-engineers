#include "popova_e_global_optimization_parallelization_by_dividing_the_search_area/seq/include/ops_seq.hpp"

#include <limits>

#include "popova_e_global_optimization_parallelization_by_dividing_the_search_area/common/include/common.hpp"

namespace popova_e_global_optimization_parallelization_by_dividing_the_search_area {

PopovaEOptimisationSEQ::PopovaEOptimisationSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = std::make_tuple(0.0, 0.0, std::numeric_limits<double>::max());
}

bool PopovaEOptimisationSEQ::ValidationImpl() {
  const auto &in = GetInput();
  // Проверяем корректность границ области и шага
  return (in.x_max > in.x_min) && (in.y_max > in.y_min) && (in.step > 0);
}

bool PopovaEOptimisationSEQ::PreProcessingImpl() {
  return true;
}

double PopovaEOptimisationSEQ::FunctionToOptimize(double x, double y) {
  const auto &in = GetInput();

  // Выбираем функцию в зависимости от func_id
  switch (in.func_id) {
    case FunctionType::kParabola1:
      // (x-2)² + (y-3)², минимум в (2, 3), значение 0
      return (x - 2.0) * (x - 2.0) + (y - 3.0) * (y - 3.0);

    case FunctionType::kParabola2:
      // x² + y², минимум в (0, 0), значение 0
      return x * x + y * y;

    case FunctionType::kParabola3:
      // (x-1)² + (y-1)² + 1, минимум в (1, 1), значение 1
      return (x - 1.0) * (x - 1.0) + (y - 1.0) * (y - 1.0) + 1.0;

    case FunctionType::kParabola4:
      // (x+1)² + (y+1)², минимум в (-1, -1), значение 0
      return (x + 1.0) * (x + 1.0) + (y + 1.0) * (y + 1.0);

    default:
      return (x - 2.0) * (x - 2.0) + (y - 3.0) * (y - 3.0);
  }
}

bool PopovaEOptimisationSEQ::RunImpl() {
  const auto &in = GetInput();

  // Инициализация: начинаем с первой точки области
  double f_min = std::numeric_limits<double>::max();
  double x_best = in.x_min;
  double y_best = in.y_min;

  // Полный перебор всех точек сетки для поиска глобального минимума
  for (double x = in.x_min; x <= in.x_max; x += in.step) {
    for (double y = in.y_min; y <= in.y_max; y += in.step) {
      double f = FunctionToOptimize(x, y);
      if (f < f_min) {
        f_min = f;
        x_best = x;
        y_best = y;
      }
    }
  }

  GetOutput() = std::make_tuple(x_best, y_best, f_min);
  return true;
}

bool PopovaEOptimisationSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace popova_e_global_optimization_parallelization_by_dividing_the_search_area
