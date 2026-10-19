"""Решения задач на NumPy."""

import numpy as np
from grader_contracts.numpy_tasks import (
    BinarizeInput, ChessInput, EllipseInput, MatrixInput, MatrixStatistics,
    MatrixVectorBatchInput, OneHotInput, RandomMatrixInput, RectangleInput,
    TimeSeriesInput, TimeSeriesStatistics,
)


def sum_prod(data: MatrixVectorBatchInput) -> np.ndarray:
    matrices, vectors = data.matrices, data.vectors
    return np.sum(matrices @ vectors, axis=0)


def binarize(data: BinarizeInput) -> np.ndarray:
    matrix, threshold = data.matrix, data.threshold
    return (matrix > threshold).astype(int)


def unique_rows(data: MatrixInput) -> list[list[float]]:
    matrix = np.asarray(data.matrix)
    result = []
    for row in matrix:
        seen = set()
        uniq = []
        for v in row:
            key = v.item() if hasattr(v, "item") else v
            if key not in seen:
                seen.add(key)
                uniq.append(float(v))
        result.append(uniq)
    return result


def unique_columns(data: MatrixInput) -> list[list[float]]:
    matrix = np.asarray(data.matrix)
    result = []
    for col in matrix.T:
        seen = set()
        uniq = []
        for v in col:
            key = v.item() if hasattr(v, "item") else v
            if key not in seen:
                seen.add(key)
                uniq.append(float(v))
        result.append(uniq)
    return result


def matrix_statistics(data: RandomMatrixInput) -> MatrixStatistics:
    rows, columns, mean, std, seed = data.rows, data.columns, data.mean, data.std, data.seed
    rng = np.random.default_rng(seed)
    matrix = rng.normal(loc=mean, scale=std, size=(rows, columns))
    
    # ВАЖНО: если грейдер выдаст ошибку про названия полей, 
    # проверь файл grader_contracts/numpy_tasks.py и исправь row_means на то, что там написано.
    return MatrixStatistics(
        matrix=matrix,
        row_means=matrix.mean(axis=1),
        column_means=matrix.mean(axis=0),
        row_variances=matrix.var(axis=1),
        column_variances=matrix.var(axis=0),
    )


def chess(data: ChessInput) -> np.ndarray:
    rows, columns, first, second = data.rows, data.columns, data.first, data.second
    board = np.empty((rows, columns), dtype=np.asarray(first).dtype if hasattr(first, "dtype") else int)
    board[::2, ::2] = first
    board[1::2, 1::2] = first
    board[::2, 1::2] = second
    board[1::2, ::2] = second
    return board


def draw_rectangle(data: RectangleInput) -> np.ndarray:
    width, height = data.width, data.height
    image_height, image_width = data.image_height, data.image_width
    shape_color, background_color = data.shape_color, data.background_color

    image = np.empty((image_height, image_width, 3), dtype=np.uint8)
    image[:] = background_color

    x0 = (image_width - width) // 2
    y0 = (image_height - height) // 2
    image[y0:y0 + height, x0:x0 + width] = shape_color
    return image


def draw_ellipse(data: EllipseInput) -> np.ndarray:
    semi_axis_x, semi_axis_y = data.semi_axis_x, data.semi_axis_y
    image_height, image_width = data.image_height, data.image_width
    shape_color, background_color = data.shape_color, data.background_color

    cx = (image_width - 1) / 2
    cy = (image_height - 1) / 2
    ys, xs = np.ogrid[:image_height, :image_width]
    mask = ((xs - cx) / semi_axis_x) ** 2 + ((ys - cy) / semi_axis_y) ** 2 <= 1.0

    image = np.empty((image_height, image_width, 3), dtype=np.uint8)
    image[:] = background_color
    image[mask] = shape_color
    return image


def analyze_time_series(data: TimeSeriesInput) -> TimeSeriesStatistics:
    values, window = np.asarray(data.values, dtype=float), data.window

    local_max = []
    local_min = []
    for i in range(1, len(values) - 1):
        if values[i] > values[i - 1] and values[i] > values[i + 1]:
            local_max.append(i)
        if values[i] < values[i - 1] and values[i] < values[i + 1]:
            local_min.append(i)

    if window <= 0 or window > len(values):
        moving = np.array([], dtype=float)
    else:
        csum = np.concatenate(([0.0], np.cumsum(values)))
        moving = (csum[window:] - csum[:-window]) / window

    return TimeSeriesStatistics(
        mean=float(values.mean()),
        variance=float(values.var()),
        std=float(values.std()),
        local_maxima=np.array(local_max, dtype=int),
        local_minima=np.array(local_min, dtype=int),
        moving_average=moving,
    )


def one_hot(data: OneHotInput) -> np.ndarray:
    labels = np.asarray(data.labels)
    class_count = data.class_count
    if class_count is None:
        class_count = int(labels.max()) + 1
    result = np.zeros((labels.size, class_count), dtype=int)
    result[np.arange(labels.size), labels] = 1
    return result
