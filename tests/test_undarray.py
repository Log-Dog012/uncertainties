import numpy as np
import pytest

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


# ---------------------------------------------------------------------------
# New tests for added functionality
# ---------------------------------------------------------------------------


def _make_und(nominals, stds):
    """Helper: create a simple UNDArray from independent values."""
    return unp.UNDArray.from_nominal_and_std(nominals, stds)


def _make_obj(nominals, stds):
    """Helper: create the equivalent object array."""
    return unp.uarray(nominals, stds)


class TestNominalAndStdDevs:
    """nominal_values() and std_devs() should return plain ndarrays for UNDArray."""

    def test_no_deprecation_warning(self):
        import warnings
        und = _make_und([1.0, 2.0], [0.1, 0.2])
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            unp.nominal_values(und)
            unp.std_devs(und)
        matrix_warns = [
            w for w in record
            if "unumpy_to_numpy_matrix" in str(w.message)
        ]
        assert matrix_warns == [], "nominal_values/std_devs called deprecated matrix helper"

    def test_returns_ndarray(self):
        und = _make_und([1.0, 2.0], [0.1, 0.2])
        assert isinstance(unp.nominal_values(und), np.ndarray)
        assert isinstance(unp.std_devs(und), np.ndarray)

    def test_values_correct(self):
        und = _make_und([3.0, 4.0], [0.3, 0.4])
        np.testing.assert_array_equal(unp.nominal_values(und), [3.0, 4.0])
        np.testing.assert_array_equal(unp.std_devs(und), [0.3, 0.4])


class TestMean:
    def test_mean_all_elements(self):
        nominals = [1.0, 2.0, 3.0, 4.0]
        stds = [0.1, 0.2, 0.1, 0.2]
        und = _make_und(nominals, stds)
        obj = _make_obj(nominals, stds)

        m_und = und.mean()
        m_obj = obj.mean()

        np.testing.assert_allclose(m_und.n, unp.nominal_values(m_obj))
        np.testing.assert_allclose(m_und.u, unp.std_devs(m_obj))

    def test_mean_axis(self):
        nominals = [[1.0, 2.0], [3.0, 4.0]]
        stds = [[0.1, 0.2], [0.1, 0.2]]
        und = _make_und(nominals, stds)
        obj = _make_obj(nominals, stds)

        m_und = und.mean(axis=0)
        m_obj = obj.mean(axis=0)

        np.testing.assert_allclose(m_und.n, unp.nominal_values(m_obj))
        np.testing.assert_allclose(m_und.u, unp.std_devs(m_obj))

    def test_mean_preserves_correlation(self):
        cov = np.array([[1.0, 0.3], [0.3, 1.0]])
        x1, x2 = uc.correlated_values([2.0, 4.0], cov)
        obj = np.array([x1, x2], dtype=object)
        und = unp.UNDArray.from_uarray(obj)

        m_und = und.mean()
        m_obj = obj.mean()

        np.testing.assert_allclose(m_und.n, float(uc.nominal_value(m_obj)))
        np.testing.assert_allclose(m_und.u, float(uc.std_dev(m_obj)))


class TestNewDerivatives:
    """Each new function derivative should match the object-array result."""

    @pytest.mark.parametrize(
        "func, nominals",
        [
            (unp.arcsin,  [0.0, 0.5]),
            (unp.arccos,  [0.0, 0.5]),
            (unp.arctan,  [0.0, 1.0]),
            (unp.asinh,   [0.0, 1.0]),
            (unp.arccosh, [1.1, 2.0]),
            (unp.arctanh, [0.0, 0.5]),
            (unp.degrees, [0.0, 1.0]),
            (unp.radians, [0.0, 90.0]),
            (unp.erf,     [0.0, 0.5]),
            (unp.erfc,    [0.0, 0.5]),
        ],
    )
    def test_unary_func_matches_object_array(self, func, nominals):
        stds = [0.1] * len(nominals)
        und = _make_und(nominals, stds)
        obj = _make_obj(nominals, stds)

        res_und = func(und)
        res_obj = func(obj)

        np.testing.assert_allclose(
            res_und.n, unp.nominal_values(res_obj), rtol=1e-12,
            err_msg=f"{func.__name__} nominal values differ",
        )
        np.testing.assert_allclose(
            res_und.u, unp.std_devs(res_obj), rtol=1e-10,
            err_msg=f"{func.__name__} std devs differ",
        )

    def test_gamma_with_scipy(self):
        """gamma() and lgamma() propagate uncertainty correctly when scipy is available."""
        pytest.importorskip("scipy")
        nominals = [1.5, 2.0]
        stds = [0.1, 0.05]
        und = _make_und(nominals, stds)
        obj = _make_obj(nominals, stds)

        res_und = unp.gamma(und)
        res_obj = unp.gamma(obj)
        np.testing.assert_allclose(res_und.n, unp.nominal_values(res_obj), rtol=1e-12)
        np.testing.assert_allclose(res_und.u, unp.std_devs(res_obj), rtol=1e-8)

        res_und = unp.lgamma(und)
        res_obj = unp.lgamma(obj)
        np.testing.assert_allclose(res_und.n, unp.nominal_values(res_obj), rtol=1e-12)
        np.testing.assert_allclose(res_und.u, unp.std_devs(res_obj), rtol=1e-8)


class TestArrayUfunc:
    """numpy ufuncs should work transparently on UNDArray via __array_ufunc__."""

    def test_np_add(self):
        a = _make_und([1.0, 2.0], [0.1, 0.2])
        b = _make_und([3.0, 4.0], [0.3, 0.4])
        res = np.add(a, b)
        assert isinstance(res, unp.UNDArray)
        np.testing.assert_allclose(res.n, [4.0, 6.0])

    def test_np_subtract(self):
        a = _make_und([5.0, 6.0], [0.1, 0.1])
        b = _make_und([1.0, 2.0], [0.1, 0.1])
        res = np.subtract(a, b)
        np.testing.assert_allclose(res.n, [4.0, 4.0])

    def test_np_multiply_scalar(self):
        a = _make_und([2.0, 3.0], [0.1, 0.2])
        res = np.multiply(a, 3.0)
        np.testing.assert_allclose(res.n, [6.0, 9.0])
        np.testing.assert_allclose(res.u, [0.3, 0.6])

    def test_np_true_divide(self):
        a = _make_und([6.0, 8.0], [0.6, 0.8])
        res = np.true_divide(a, 2.0)
        np.testing.assert_allclose(res.n, [3.0, 4.0])
        np.testing.assert_allclose(res.u, [0.3, 0.4])

    def test_np_sin(self):
        nominals = [0.0, np.pi / 4]
        stds = [0.1, 0.1]
        und = _make_und(nominals, stds)
        obj = _make_obj(nominals, stds)
        res = np.sin(und)
        np.testing.assert_allclose(res.n, unp.nominal_values(unp.sin(obj)), rtol=1e-12)
        np.testing.assert_allclose(res.u, unp.std_devs(unp.sin(obj)), rtol=1e-10)

    def test_np_sqrt(self):
        und = _make_und([4.0, 9.0], [0.4, 0.9])
        res = np.sqrt(und)
        np.testing.assert_allclose(res.n, [2.0, 3.0])
        np.testing.assert_allclose(res.u, [0.1, 0.15], rtol=1e-10)

    def test_np_exp_log_roundtrip(self):
        und = _make_und([1.0, 2.0], [0.1, 0.2])
        res = np.log(np.exp(und))
        np.testing.assert_allclose(res.n, und.n, rtol=1e-12)
        np.testing.assert_allclose(res.u, und.u, rtol=1e-10)

    def test_np_power(self):
        und = _make_und([2.0, 3.0], [0.2, 0.3])
        res = np.power(und, 2)
        np.testing.assert_allclose(res.n, [4.0, 9.0])
        np.testing.assert_allclose(res.u, [0.8, 1.8], rtol=1e-10)

    def test_np_negative(self):
        und = _make_und([1.0, -2.0], [0.1, 0.2])
        res = np.negative(und)
        np.testing.assert_allclose(res.n, [-1.0, 2.0])
        np.testing.assert_allclose(res.u, [0.1, 0.2])

    def test_np_absolute(self):
        und = _make_und([-3.0, 4.0], [0.3, 0.4])
        res = np.absolute(und)
        np.testing.assert_allclose(res.n, [3.0, 4.0])

    def test_np_add_reflected(self):
        """scalar + UNDArray should work via reflected ufunc."""
        und = _make_und([1.0, 2.0], [0.1, 0.2])
        res = np.add(5.0, und)
        np.testing.assert_allclose(res.n, [6.0, 7.0])
        np.testing.assert_allclose(res.u, [0.1, 0.2])

    def test_ufunc_preserves_correlation(self):
        """np.add on correlated UNDArrays should preserve full covariance."""
        cov = np.array([[1.0, 0.5], [0.5, 1.0]])
        x1, x2 = uc.correlated_values([1.0, 2.0], cov)
        obj = np.array([x1, x2], dtype=object)
        und = unp.UNDArray.from_uarray(obj)

        res_obj = np.add(obj, obj)
        res_und = np.add(und, und)

        np.testing.assert_allclose(res_und.n, unp.nominal_values(res_obj))
        np.testing.assert_allclose(res_und.u, unp.std_devs(res_obj))
        cov_obj = uc.covariance_matrix(res_obj)
        cov_und = uc.covariance_matrix(res_und.to_uarray())
        np.testing.assert_allclose(cov_obj, cov_und)


class TestBinaryFunctions:
    def test_hypot_matches_object_array(self):
        a = _make_und([3.0, 4.0], [0.3, 0.4])
        b = _make_und([4.0, 5.0], [0.2, 0.1])
        a_obj = _make_obj([3.0, 4.0], [0.3, 0.4])
        b_obj = _make_obj([4.0, 5.0], [0.2, 0.1])

        res_obj = unp.hypot(a_obj, b_obj)
        res_unp = unp.hypot(a, b)
        res_np = np.hypot(a, b)

        np.testing.assert_allclose(res_unp.n, unp.nominal_values(res_obj))
        np.testing.assert_allclose(res_unp.u, unp.std_devs(res_obj), rtol=1e-10)
        np.testing.assert_allclose(res_np.n, unp.nominal_values(res_obj))
        np.testing.assert_allclose(res_np.u, unp.std_devs(res_obj), rtol=1e-10)

    def test_arctan2_matches_object_array(self):
        y = _make_und([1.0, 2.0], [0.1, 0.2])
        x = _make_und([2.0, 1.5], [0.2, 0.1])
        y_obj = _make_obj([1.0, 2.0], [0.1, 0.2])
        x_obj = _make_obj([2.0, 1.5], [0.2, 0.1])

        res_obj = unp.arctan2(y_obj, x_obj)
        res_unp = unp.arctan2(y, x)
        res_np = np.arctan2(y, x)

        np.testing.assert_allclose(res_unp.n, unp.nominal_values(res_obj))
        np.testing.assert_allclose(res_unp.u, unp.std_devs(res_obj), rtol=1e-10)
        np.testing.assert_allclose(res_np.n, unp.nominal_values(res_obj))
        np.testing.assert_allclose(res_np.u, unp.std_devs(res_obj), rtol=1e-10)

    def test_power_with_uncertain_exponent(self):
        base = _make_und([2.0, 3.0], [0.2, 0.3])
        exponent = _make_und([1.5, 2.0], [0.1, 0.05])
        base_obj = _make_obj([2.0, 3.0], [0.2, 0.3])
        exponent_obj = _make_obj([1.5, 2.0], [0.1, 0.05])

        res_obj = np.power(base_obj, exponent_obj)
        res_np = np.power(base, exponent)
        res_unp = unp.pow(base, exponent)

        np.testing.assert_allclose(res_np.n, unp.nominal_values(res_obj))
        np.testing.assert_allclose(res_np.u, unp.std_devs(res_obj), rtol=1e-10)
        np.testing.assert_allclose(res_unp.n, unp.nominal_values(res_obj))
        np.testing.assert_allclose(res_unp.u, unp.std_devs(res_obj), rtol=1e-10)

        res_obj_rpow = 2.0 ** exponent_obj
        res_rpow = 2.0 ** exponent
        np.testing.assert_allclose(res_rpow.n, unp.nominal_values(res_obj_rpow))
        np.testing.assert_allclose(res_rpow.u, unp.std_devs(res_obj_rpow), rtol=1e-10)
