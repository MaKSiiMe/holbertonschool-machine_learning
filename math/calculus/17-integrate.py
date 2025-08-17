#!/usr/bin/env python3

def poly_integral(poly, C=0):
    if (not isinstance(poly, list) or
        len(poly) == 0 or
        not all(isinstance(c, (int, float)) for c in poly) or
        not isinstance(C, int)):
        return None
    integral = [C]
    for i in range(len(poly)):
        val = poly[i] / (i + 1)
        if val == int(val):
            val = int(val)
        integral.append(val)
    while len(integral) > 1 and integral[-1] == 0:
        integral.pop()
    return integral
