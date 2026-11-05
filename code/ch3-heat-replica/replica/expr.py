"""Evaluate the COMSOL expressions the model uses (parameters, units, material functions)."""
import re

import numpy as np

# Conversion to SI of the bracketed units that appear in the model.
UNIT = {
    "1": 1.0, "K": 1.0, "m": 1.0, "s": 1.0, "Pa": 1.0, "mm": 1e-3, "um": 1e-6, "min": 60.0,
    "atm": 101325.0, "W/(m*K)": 1.0, "kg/m^3": 1.0, "J/(kg*K)": 1.0, "W/m^3": 1.0, "W/m^2": 1.0,
    "1/s": 1.0, "Pa*s": 1.0, "m/s": 1.0, "K*mol/J": 1.0,
}

CONST = {
    "R_const": 8.31446261815324,       # J/(mol K)
    "g_const": 9.80665,                # m/s^2
    "sigma_const": 5.670374419e-8,     # W/(m^2 K^4)
}


def to_python(text):
    """COMSOL expression -> Python expression (units become SI factors, ^ becomes **)."""
    def unit(m):
        u = m.group(1).replace(" ", "")
        if u not in UNIT:
            raise ValueError(f"unit [{u}] is not in the conversion table")
        return f"*{UNIT[u]!r}"
    s = re.sub(r"\s*\[([^\]]+)\]", unit, text)
    return s.replace("^", "**")


def smooth_step(location, smooth):
    """COMSOL Step function from 0 to 1 with two continuous derivatives.

    The transition is the fifth-order polynomial COMSOL uses (flc2hs) over a
    zone of total width ``smooth`` centred on ``location``.
    """
    half = smooth / 2.0

    def f(x):
        s = np.clip((np.asarray(x, float) - location) / half, -1.0, 1.0)
        return 0.5 + 0.9375 * s - 0.625 * s ** 3 + 0.1875 * s ** 5
    return f


def piecewise(arg, pieces, scope):
    """Piecewise function of one argument; constant extrapolation outside the table."""
    code = [(lo, hi, compile(to_python(e), "<piecewise>", "eval")) for lo, hi, e in pieces]
    lo0, hi0 = pieces[0][0], pieces[-1][1]

    def f(x):
        x = np.clip(np.asarray(x, float), lo0, np.nextafter(hi0, lo0))
        out = np.empty_like(x)
        for lo, hi, c in code:
            m = (x >= lo) & (x < hi)
            if m.any():
                out[m] = eval(c, scope, {arg: x[m]})
        return out
    return f


def analytic(expr, args, scope):
    code = compile(to_python(expr), "<analytic>", "eval")

    def f(*a):
        return eval(code, scope, dict(zip(args, a)))
    return f


class Scope:
    """Namespace for evaluating expressions: parameters, constants and functions."""

    def __init__(self, parameters, functions=None):
        self.names = {"__builtins__": {}, "exp": np.exp, "log": np.log, "sqrt": np.sqrt, "abs": np.abs,
                      "pi": np.pi}
        self.names.update(CONST)
        self.names.update(parameters)
        if functions:
            self.names.update(functions)

    def child(self, funcs):
        """Scope extended with material functions given as definitions from the reader."""
        s = Scope({}, None)
        s.names = dict(self.names)
        for name, d in funcs.items():
            if d["kind"] == "piecewise":
                s.names[name] = piecewise(d["arg"], d["pieces"], s.names)
            else:
                s.names[name] = analytic(d["expr"], d["args"], s.names)
        return s

    def eval(self, text, **local):
        return eval(compile(to_python(text), "<expr>", "eval"), self.names, local)

    def uses(self, text, name):
        return re.search(rf"(?<![A-Za-z0-9_]){name}(?![A-Za-z0-9_])", text) is not None
