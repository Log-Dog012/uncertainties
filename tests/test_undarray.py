import numpy as np

import uncertainties.core as uc
import uncertainties.unumpy as unp


def test_undarray_matches_object_array_for_unary_func_with_correlation():
    cov = np.array([[1.0, 0.2], [0.2, 4.0]])
    x1, x2 = uc.correlated_values([1.0, 2.0], cov)
    obj = np.array([x1, x2], dtype=object)

    und = unp.UNDArray.from_uarray(obj)

    res_obj = unp.sin(obj)
    res_und = unp.sin(und)

    np.testing.assert_allclose(unp.nominal_values(res_obj), unp.nominal_values(res_und))
    np.testing.assert_allclose(unp.std_devs(res_obj), unp.std_devs(res_und))

    cov_obj = uc.covariance_matrix(res_obj)
    cov_und = uc.covariance_matrix(res_und.to_uarray())
    np.testing.assert_allclose(cov_obj, cov_und)


def test_undarray_binary_ops_preserve_correlation():
    cov = np.array([[1.0, 0.5], [0.5, 1.0]])
    x1, x2 = uc.correlated_values([3.0, -1.0], cov)
    y1, y2 = uc.correlated_values([0.5, 2.0], cov)

    a_obj = np.array([x1, x2], dtype=object)
    b_obj = np.array([y1, y2], dtype=object)

    a = unp.UNDArray.from_uarray(a_obj)
    b = unp.UNDArray.from_uarray(b_obj)

    res_obj = (a_obj * 2.0 + b_obj) / 3.0
    res = (a * 2.0 + b) / 3.0

    np.testing.assert_allclose(unp.nominal_values(res_obj), res.n)
    np.testing.assert_allclose(unp.std_devs(res_obj), res.u)

    cov_obj = uc.covariance_matrix(res_obj)
    cov_new = uc.covariance_matrix(res.to_uarray())
    np.testing.assert_allclose(cov_obj, cov_new)


def test_undarray_sum_matches_object_array():
    cov = np.array([[1.0, 0.1], [0.1, 2.0]])
    x1, x2 = uc.correlated_values([1.0, 2.0], cov)
    obj = np.array([[x1, x2], [x2, x1]], dtype=object)

    und = unp.UNDArray.from_uarray(obj)
    s_obj = obj.sum(axis=0)
    s_und = und.sum(axis=0)

    np.testing.assert_allclose(unp.nominal_values(s_obj), s_und.n)
    np.testing.assert_allclose(unp.std_devs(s_obj), s_und.u)

