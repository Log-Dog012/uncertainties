"""uncertainties.unumpy.UNDArray

An array container for affine (first-order) uncertainty propagation.

The goal is to keep bulk access to nominal values/standard deviations
while preserving correlation information through a shared linear part.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Any, Callable

import numpy as np

import uncertainties.core as uncert_core


def _as_float64_ndarray(values: Any) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    return arr


def _broadcast_to_shape(values: Any, shape: tuple[int, ...]) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.shape == shape:
        return arr
    return np.broadcast_to(arr, shape).astype(np.float64, copy=False)


@dataclass(frozen=True)
class _LinearPart:
    """Linear part for an UNDArray.

    Each element is represented as a mapping {Variable: coefficient}.
    This matches the structure used by uncertainties.core.LinearCombination.
    """

    coeffs: np.ndarray  # dtype=object, elements are dict[Variable, float]

    @staticmethod
    def zeros(shape: tuple[int, ...]) -> "_LinearPart":
        coeffs = np.empty(shape, dtype=object)
        coeffs.flat[:] = ({},)
        return _LinearPart(coeffs)

    def scale(self, factor: np.ndarray) -> "_LinearPart":
        factor = np.asarray(factor, dtype=np.float64)
        if factor.shape != self.coeffs.shape:
            factor = np.broadcast_to(factor, self.coeffs.shape)

        out = np.empty(self.coeffs.shape, dtype=object)

        def scale_one(d: dict[uncert_core.Variable, float], f: float):
            if not d or f == 0.0:
                return {}
            return {var: coeff * f for var, coeff in d.items() if coeff * f != 0.0}

        for i, (d, f) in enumerate(zip(self.coeffs.flat, factor.flat)):
            out.flat[i] = scale_one(d, float(f))
        return _LinearPart(out)

    def add(self, other: "_LinearPart") -> "_LinearPart":
        if self.coeffs.shape != other.coeffs.shape:
            raise ValueError("Linear parts must have identical shapes")

        out = np.empty(self.coeffs.shape, dtype=object)

        def add_one(a: dict[uncert_core.Variable, float], b: dict[uncert_core.Variable, float]):
            if not a:
                return dict(b)
            if not b:
                return dict(a)
            merged: dict[uncert_core.Variable, float] = dict(a)
            for var, coeff in b.items():
                merged[var] = merged.get(var, 0.0) + coeff
                if merged[var] == 0.0:
                    merged.pop(var, None)
            return merged

        for i, (a, b) in enumerate(zip(self.coeffs.flat, other.coeffs.flat)):
            out.flat[i] = add_one(a, b)
        return _LinearPart(out)


class UNDArray:
    """Correlation-aware array of affine uncertainty quantities.

    Parameters
    ----------
    nominal_values:
        Array-like of nominal values.
    linear_part:
        An ndarray (dtype=object) with the same shape as nominal_values,
        whose elements are dicts mapping Variable -> coefficient.
    """

    __slots__ = ("n", "_linear")

    def __init__(self, nominal_values: Any, linear_part: _LinearPart):
        self.n = _as_float64_ndarray(nominal_values)
        if linear_part.coeffs.shape != self.n.shape:
            raise ValueError("linear_part shape must match nominal_values")
        self._linear = linear_part

    @property
    def shape(self) -> tuple[int, ...]:
        return self.n.shape

    @property
    def ndim(self) -> int:
        return self.n.ndim

    @property
    def size(self) -> int:
        return self.n.size

    def __array__(self, dtype=None):
        return np.asarray(self.n, dtype=dtype)

    def __repr__(self) -> str:
        return f"UNDArray(n={self.n!r}, u={self.u!r})"

    @property
    def u(self) -> np.ndarray:
        """Standard deviations, derived from the current Variable std devs."""

        # Fast path for constants:
        if self.size == 0:
            return np.asarray(self.n, dtype=np.float64)

        out = np.zeros(self.shape, dtype=np.float64)
        for i, d in enumerate(self._linear.coeffs.flat):
            if not d:
                continue
            var_sum = 0.0
            for var, coeff in d.items():
                # Variable._std_dev can change over time.
                var_sum += (coeff * var._std_dev) ** 2
            out.flat[i] = sqrt(var_sum)
        return out

    def std_devs(self) -> np.ndarray:
        return self.u

    def nominal_values(self) -> np.ndarray:
        return self.n

    def to_uarray(self) -> np.ndarray:
        """Convert to a numpy object array of AffineScalarFunc."""

        out = np.empty(self.shape, dtype=object)
        for i, (nom, d) in enumerate(zip(self.n.flat, self._linear.coeffs.flat)):
            out.flat[i] = uncert_core.AffineScalarFunc(float(nom), uncert_core.LinearCombination(d))
        return out

    @staticmethod
    def from_nominal_and_std(nominal_values: Any, std_devs: Any, tag: Any = None) -> "UNDArray":
        n = _as_float64_ndarray(nominal_values)
        u = _broadcast_to_shape(std_devs, n.shape)
        coeffs = np.empty(n.shape, dtype=object)
        for i, (nom, sd) in enumerate(zip(n.flat, u.flat)):
            var = uncert_core.Variable(float(nom), float(sd), tag)
            coeffs.flat[i] = {var: 1.0} if sd != 0.0 else {}
        return UNDArray(n, _LinearPart(coeffs))

    @staticmethod
    def from_uarray(arr: Any) -> "UNDArray":
        """Build from an array-like of uncertainties objects.

        Preserves correlation by reusing the same Variable objects.
        """

        obj = np.asarray(arr, dtype=object)
        n = np.vectorize(uncert_core.nominal_value, otypes=[float])(obj)

        coeffs = np.empty(obj.shape, dtype=object)
        for i, element in enumerate(obj.flat):
            if isinstance(element, uncert_core.AffineScalarFunc):
                lp = element._linear_part
                if not lp.expanded():
                    lp.expand()
                coeffs.flat[i] = dict(lp.linear_combo)
            else:
                coeffs.flat[i] = {}
        return UNDArray(n, _LinearPart(coeffs))

    def _unary_op(self, func: Callable[[np.ndarray], np.ndarray], deriv: Callable[[np.ndarray], np.ndarray]) -> "UNDArray":
        n_out = func(self.n)
        factor = deriv(self.n)
        linear_out = self._linear.scale(factor)
        return UNDArray(n_out, linear_out)

    def _binary_op(
        self,
        other: Any,
        func: Callable[[np.ndarray, np.ndarray], np.ndarray],
        dself: Callable[[np.ndarray, np.ndarray], np.ndarray],
        dother: Callable[[np.ndarray, np.ndarray], np.ndarray],
    ) -> "UNDArray":
        other_arr = other
        if isinstance(other, UNDArray):
            n_other = other.n
            l_other = other._linear
        else:
            n_other = _as_float64_ndarray(other)
            n_other = np.broadcast_to(n_other, np.broadcast(self.n, n_other).shape)
            l_other = _LinearPart.zeros(n_other.shape)

        n_self = np.broadcast_to(self.n, np.broadcast(self.n, n_other).shape)
        if n_self.shape != self.n.shape:
            l_self = _LinearPart(np.broadcast_to(self._linear.coeffs, n_self.shape))
        else:
            l_self = self._linear

        if n_other.shape != n_self.shape:
            n_other = np.broadcast_to(n_other, n_self.shape)
        if l_other.coeffs.shape != n_self.shape:
            l_other = _LinearPart(np.broadcast_to(l_other.coeffs, n_self.shape))

        n_out = func(n_self, n_other)
        l_out = l_self.scale(dself(n_self, n_other)).add(l_other.scale(dother(n_self, n_other)))
        return UNDArray(n_out, l_out)

    def __neg__(self) -> "UNDArray":
        return UNDArray(-self.n, self._linear.scale(-1.0))

    def __add__(self, other: Any) -> "UNDArray":
        return self._binary_op(other, lambda a, b: a + b, lambda a, b: 1.0, lambda a, b: 1.0)

    def __radd__(self, other: Any) -> "UNDArray":
        return self.__add__(other)

    def __sub__(self, other: Any) -> "UNDArray":
        return self._binary_op(other, lambda a, b: a - b, lambda a, b: 1.0, lambda a, b: -1.0)

    def __rsub__(self, other: Any) -> "UNDArray":
        return (-self).__add__(other)

    def __mul__(self, other: Any) -> "UNDArray":
        return self._binary_op(other, lambda a, b: a * b, lambda a, b: b, lambda a, b: a)

    def __rmul__(self, other: Any) -> "UNDArray":
        return self.__mul__(other)

    def __truediv__(self, other: Any) -> "UNDArray":
        return self._binary_op(
            other,
            lambda a, b: a / b,
            lambda a, b: 1.0 / b,
            lambda a, b: -a / (b * b),
        )

    def __rtruediv__(self, other: Any) -> "UNDArray":
        # other / self
        return UNDArray.from_nominal_and_std(other, 0.0)._binary_op(
            self,
            lambda a, b: a / b,
            lambda a, b: 1.0 / b,
            lambda a, b: -a / (b * b),
        )

    def __pow__(self, other: Any) -> "UNDArray":
        # General pow with both operands uncertain is tricky; for now,
        # support scalar/ndarray exponent without uncertainty.
        if isinstance(other, UNDArray):
            raise TypeError("UNDArray exponentiation only supports non-UNDArray exponents")
        p = _as_float64_ndarray(other)
        p = np.broadcast_to(p, np.broadcast(self.n, p).shape)
        n_self = np.broadcast_to(self.n, p.shape)
        l_self = self._linear
        if n_self.shape != self.n.shape:
            l_self = _LinearPart(np.broadcast_to(self._linear.coeffs, n_self.shape))
        n_out = n_self**p
        factor = p * (n_self ** (p - 1.0))
        return UNDArray(n_out, l_self.scale(factor))

    def reshape(self, *shape: int) -> "UNDArray":
        n = self.n.reshape(*shape)
        coeffs = self._linear.coeffs.reshape(*shape)
        return UNDArray(n, _LinearPart(coeffs))

    def __getitem__(self, item) -> "UNDArray":
        n = self.n[item]
        coeffs = self._linear.coeffs[item]
        return UNDArray(n, _LinearPart(np.asarray(coeffs, dtype=object)))

    def sum(self, axis=None, dtype=None, out=None, keepdims=False):
        if out is not None:
            raise TypeError("out= is not supported for UNDArray.sum")
        n_out = self.n.sum(axis=axis, dtype=dtype, keepdims=keepdims)
        coeffs = self._linear.coeffs
        if axis is None:
            merged: dict[uncert_core.Variable, float] = {}
            for d in coeffs.flat:
                for var, c in d.items():
                    merged[var] = merged.get(var, 0.0) + c
            out_coeffs = np.asarray(merged, dtype=object)
            # Scalar result shape is ()
            out_arr = np.empty((), dtype=object)
            out_arr[()] = merged
            return UNDArray(np.asarray(n_out), _LinearPart(out_arr))

        # Reduce along an axis: build dict per output element.
        coeffs_moved = np.moveaxis(coeffs, axis, 0)
        out_shape = coeffs_moved.shape[1:]
        out_coeffs = np.empty(out_shape, dtype=object)
        for idx in np.ndindex(out_shape):
            merged: dict[uncert_core.Variable, float] = {}
            for d in coeffs_moved[(slice(None),) + idx]:
                for var, c in d.items():
                    merged[var] = merged.get(var, 0.0) + c
            out_coeffs[idx] = merged
        return UNDArray(np.asarray(n_out), _LinearPart(out_coeffs))
