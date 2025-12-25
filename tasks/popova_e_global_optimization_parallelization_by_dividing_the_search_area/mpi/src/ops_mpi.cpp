#include "popova_e_global_optimization_parallelization_by_dividing_the_search_area/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <limits>

#include "popova_e_global_optimization_parallelization_by_dividing_the_search_area/common/include/common.hpp"

namespace popova_e_global_optimization_parallelization_by_dividing_the_search_area {

PopovaEGlobalOptimizationDividingSearchAreaMPI::PopovaEGlobalOptimizationDividingSearchAreaMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = std::make_tuple(0.0, 0.0, std::numeric_limits<double>::max());
}

bool PopovaEGlobalOptimizationDividingSearchAreaMPI::ValidationImpl() {
  const auto &in = GetInput();
  // Проверяем корректность границ области и шага
  return (in.x_max > in.x_min) && (in.y_max > in.y_min) && (in.step > 0);
}

bool PopovaEGlobalOptimizationDividingSearchAreaMPI::PreProcessingImpl() {
  return true;
}

double PopovaEGlobalOptimizationDividingSearchAreaMPI::FunctionToOptimize(double x, double y) {
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

void PopovaEGlobalOptimizationDividingSearchAreaMPI::ComputeProcessArea(int rank, int size, double &x_start,
                                                                        double &x_end, double &y_start, double &y_end) {
  const auto &in = GetInput();

  // Разбиваем процессы на 2D сетку (близко к квадрату для лучшей балансировки)
  int rows = static_cast<int>(std::sqrt(static_cast<double>(size)));
  while (size % rows != 0 && rows > 0) {
    --rows;
  }
  if (rows == 0) {
    rows = 1;
  }
  int cols = size / rows;

  // Определяем позицию текущего процесса в сетке
  int row = rank / cols;
  int col = rank % cols;

  // Вычисляем размеры под-области для этого процесса
  double x_range = in.x_max - in.x_min;
  double y_range = in.y_max - in.y_min;
  double x_step = x_range / cols;
  double y_step = y_range / rows;

  // Вычисляем границы области для этого процесса
  x_start = in.x_min + col * x_step;
  if (col == cols - 1) {
    x_end = in.x_max;
  } else {
    x_end = in.x_min + (col + 1) * x_step;
  }

  y_start = in.y_min + row * y_step;
  if (row == rows - 1) {
    y_end = in.y_max;
  } else {
    y_end = in.y_min + (row + 1) * y_step;
  }
}

void PopovaEGlobalOptimizationDividingSearchAreaMPI::CoarseSearch(double x_start, double x_end, double y_start,
                                                                  double y_end, double big_step, double &best_x,
                                                                  double &best_y, double &min_value) {
  // Инициализация: начинаем с первой точки области
  min_value = std::numeric_limits<double>::max();
  best_x = x_start;
  best_y = y_start;

  // Грубый поиск с большим шагом для быстрого сканирования области
  for (double x = x_start; x <= x_end; x += big_step) {
    for (double y = y_start; y <= y_end; y += big_step) {
      double value = FunctionToOptimize(x, y);
      if (value < min_value) {
        min_value = value;
        best_x = x;
        best_y = y;
      }
    }
  }
}

void PopovaEGlobalOptimizationDividingSearchAreaMPI::FineSearch(double x_start, double x_end, double y_start,
                                                                double y_end, double step, double &best_x,
                                                                double &best_y, double &min_value) {
  // Точный поиск с заданным шагом в указанной области
  for (double x = x_start; x <= x_end; x += step) {
    for (double y = y_start; y <= y_end; y += step) {
      double value = FunctionToOptimize(x, y);
      if (value < min_value) {
        min_value = value;
        best_x = x;
        best_y = y;
      }
    }
  }
}

void PopovaEGlobalOptimizationDividingSearchAreaMPI::SearchRemainingArea(double x_start, double x_end, double y_start,
                                                                         double y_end, double skip_x_start,
                                                                         double skip_x_end, double skip_y_start,
                                                                         double skip_y_end, double middle_step,
                                                                         double fine_step, double &best_x,
                                                                         double &best_y, double &min_value) {
  // Поиск в остальной области с промежуточным шагом для обнаружения других локальных минимумов
  for (double x = x_start; x <= x_end; x += middle_step) {
    for (double y = y_start; y <= y_end; y += middle_step) {
      // Пропускаем уже проверенную область
      if (x >= skip_x_start && x <= skip_x_end && y >= skip_y_start && y <= skip_y_end) {
        continue;
      }

      double value = FunctionToOptimize(x, y);
      // Если найдена лучшая точка, уточняем поиск в её окрестности
      if (value < min_value) {
        double radius = 2.0 * middle_step;
        double local_x_start = std::max(x_start, x - radius);
        double local_x_end = std::min(x_end, x + radius);
        double local_y_start = std::max(y_start, y - radius);
        double local_y_end = std::min(y_end, y + radius);

        FineSearch(local_x_start, local_x_end, local_y_start, local_y_end, fine_step, best_x, best_y, min_value);
      }
    }
  }
}

void PopovaEGlobalOptimizationDividingSearchAreaMPI::FindGlobalMinimum(double local_x, double local_y,
                                                                       double local_value, double &global_x,
                                                                       double &global_y, double &global_value) {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  // Структура для передачи значения и ранга процесса
  struct {
    double value;
    int rank;
  } local_min, global_min;

  local_min.value = local_value;
  local_min.rank = rank;

  // Находим процесс с минимальным значением среди всех процессов
  MPI_Allreduce(&local_min, &global_min, 1, MPI_DOUBLE_INT, MPI_MINLOC, MPI_COMM_WORLD);

  // Процесс с глобальным минимумом отправляет координаты всем остальным
  double coords[2] = {local_x, local_y};
  MPI_Bcast(coords, 2, MPI_DOUBLE, global_min.rank, MPI_COMM_WORLD);

  global_x = coords[0];
  global_y = coords[1];
  global_value = global_min.value;
}

bool PopovaEGlobalOptimizationDividingSearchAreaMPI::RunImpl() {
  const auto &in = GetInput();

  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  // Разделяем область поиска между процессами в 2D сетке
  double x_start, x_end, y_start, y_end;
  ComputeProcessArea(rank, size, x_start, x_end, y_start, y_end);

  // Многошаговая схема оптимизации:
  // Шаг 1: Грубый поиск с большим шагом для быстрого определения перспективной области
  double big_step = in.step * 5.0;
  double best_x, best_y, min_value;
  CoarseSearch(x_start, x_end, y_start, y_end, big_step, best_x, best_y, min_value);

  // Шаг 2: Точный поиск в окрестности найденной точки
  double search_radius = 3.0 * big_step;
  double refine_x_start = std::max(x_start, best_x - search_radius);
  double refine_x_end = std::min(x_end, best_x + search_radius);
  double refine_y_start = std::max(y_start, best_y - search_radius);
  double refine_y_end = std::min(y_end, best_y + search_radius);

  FineSearch(refine_x_start, refine_x_end, refine_y_start, refine_y_end, in.step, best_x, best_y, min_value);

  // Шаг 3: Поиск в остальной области для обнаружения других локальных минимумов
  double middle_step = in.step * 3.0;
  SearchRemainingArea(x_start, x_end, y_start, y_end, refine_x_start, refine_x_end, refine_y_start, refine_y_end,
                      middle_step, in.step, best_x, best_y, min_value);

  // Находим глобальный минимум среди всех процессов через MPI
  double global_x, global_y, global_value;
  FindGlobalMinimum(best_x, best_y, min_value, global_x, global_y, global_value);

  GetOutput() = std::make_tuple(global_x, global_y, global_value);

  return true;
}

bool PopovaEGlobalOptimizationDividingSearchAreaMPI::PostProcessingImpl() {
  return true;
}

}  // namespace popova_e_global_optimization_parallelization_by_dividing_the_search_area
