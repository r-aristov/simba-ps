from PIL import Image
import numpy as np
from numba import cuda

from simba_ps.utils.util_kernels import get_bpg, select_particle_kernel, set_vector_for_p_type
from simba_ps.visualization.visualization_kernels import set_colors_in_rect_kernel


def set_vector_by_p_type(p_type_to_set, vector_value, particle_count, p_types, vectors):
    threads_per_block = (256, 2)
    blocks_per_grid = get_bpg(threads_per_block,
                              particle_count,
                              2)
    set_vector_for_p_type[blocks_per_grid, threads_per_block](p_types, vectors, p_type_to_set, vector_value)


def load_img_to_gpu(filename):
    img = Image.open(filename)
    img = np.array(img, dtype=float)
    img = img / 255.0
    return cuda.to_device(img)


def set_colors_in_rect(colors, image, p_types, positions, cx, cy, p_types_to_set=None):
    p_types_to_set = cuda.to_device(np.array([-1])) if p_types_to_set is None else cuda.to_device(
        np.array(list(p_types_to_set)))

    threads_per_block = (10, 3)
    blocks_per_grid = get_bpg(threads_per_block, positions.shape[0], 3)
    set_colors_in_rect_kernel[blocks_per_grid, threads_per_block](colors, image, p_types, positions, cx, cy,
                                                                  p_types_to_set)


def select_particle(positions, x, y):
    output = cuda.to_device(np.array([-1], dtype=int))
    threads_per_block = (16,)
    blocks_per_grid = get_bpg(threads_per_block, positions.shape[0])
    select_particle_kernel[blocks_per_grid, threads_per_block](positions, x, y, output)
    return output[0]
