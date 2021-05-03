import numpy as np

from summer.compute import binary_matrix_to_sparse_pairs, sparse_pairs_accum


def test_binary_matrix_to_pairs():
    n_rows = 4
    n_cols = 10

    np.random.seed(0)

    category_matrix = np.random.choice([0.0, 1.0], size=(n_rows, n_cols))

    idx_mapping = binary_matrix_to_sparse_pairs(category_matrix)

    assert len(idx_mapping) == (category_matrix == 1).sum()

    regen_idx = np.zeros_like(category_matrix)
    for src, target in idx_mapping:
        regen_idx[target, src] = 1

    assert (regen_idx == category_matrix).all()


def test_sparse_pairs_accum_equals_matmul():
    n_rows = 16
    n_cols = 1000

    # Make sure this works for a variety of values
    for seed in range(10):
        np.random.seed(seed)

        category_matrix = np.random.choice([0.0, 1.0], size=(n_rows, n_cols))

        # Ensure wide range of values
        comp_vals = np.random.uniform(high=1e10, size=(n_cols))

        idx_mapping = binary_matrix_to_sparse_pairs(category_matrix)
        sp_res = sparse_pairs_accum(idx_mapping, comp_vals, n_rows)
        mm_res = np.matmul(category_matrix, comp_vals)

        np.testing.assert_allclose(mm_res, sp_res)


def test_sparse_pairs_accum_equals_sum():
    n_rows = 16
    n_cols = 1000

    # Make sure this works for a variety of values
    for seed in range(10):
        np.random.seed(seed)

        category_matrix = np.random.choice([0.0, 1.0], size=(n_rows, n_cols))

        # Ensure wide range of values
        comp_vals = np.random.uniform(high=1e10, size=(n_cols))

        idx_mapping = binary_matrix_to_sparse_pairs(category_matrix)
        sp_res = sparse_pairs_accum(idx_mapping, comp_vals, n_rows)

        sum_vals = np.empty(n_rows)
        for i in range(n_rows):
            row_target_mask = idx_mapping[:, 1] == i
            sum_vals[i] = sum(comp_vals[idx_mapping[:, 0][row_target_mask]])

        np.testing.assert_allclose(sp_res, sum_vals)
