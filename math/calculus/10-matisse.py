#!/usr/bin/env python3

def poly_derivative(poly):
    if (not isinstance(poly, list) or
        len(poly) == 0 or
        not all(isinstance(c, (int, float)) for c in poly)):
        return None
    if len(poly) == 1:
        return [0]
    deriv = [i * coef for i, coef in enumerate(poly)][1:]
    if all(c == 0 for c in deriv):
        return [0]
    return deriv
