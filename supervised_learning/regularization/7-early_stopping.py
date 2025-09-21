#!/usr/bin/env python3
"""7. Early Stopping"""


def early_stopping(cost, opt_cost, threshold, patience, count):
    """Determines if you should stop gradient descent early"""
    if cost <= opt_cost - threshold:
        return False, 0
    else:
        count += 1

        if count >= patience:
            return True, count
        else:
            return False, count
