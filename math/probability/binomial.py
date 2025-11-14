#!/usr/bin/env python3
"""Module for Binomial distribution"""


class Binomial:
    """Represents a Binomial distribution"""

    def __init__(self, data=None, n=1, p=0.5):
        """
        Initialize Binomial distribution

        Args:
            data: list of data to estimate the distribution
            n: number of Bernoulli trials
            p: probability of a success
        """
        if data is None:
            if n <= 0:
                raise ValueError("n must be a positive value")
            if p <= 0 or p >= 1:
                raise ValueError("p must be greater than 0 and less than 1")
            self.n = int(n)
            self.p = float(p)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")

            # Calculate mean and variance
            mean = sum(data) / len(data)
            variance = sum((x - mean) ** 2 for x in data) / len(data)

            # For binomial: variance = n*p*(1-p) and mean = n*p
            # So: p = 1 - (variance/mean)
            # And: n = mean/p
            p_estimate = 1 - (variance / mean)
            n_estimate = mean / p_estimate

            # Round n to nearest integer
            self.n = round(n_estimate)

            # Recalculate p with rounded n for better accuracy
            self.p = float(mean / self.n)

    def pmf(self, k):
        """
        Calculates the value of the PMF for a given number of successes

        Args:
            k: number of successes

        Returns:
            PMF value for k
        """
        k = int(k)
        if k < 0 or k > self.n:
            return 0

        # Calculate binomial coefficient C(n,k) = n! / (k! * (n-k)!)
        n_factorial = 1
        for i in range(1, self.n + 1):
            n_factorial *= i

        k_factorial = 1
        for i in range(1, k + 1):
            k_factorial *= i

        nk_factorial = 1
        for i in range(1, self.n - k + 1):
            nk_factorial *= i

        binomial_coeff = n_factorial / (k_factorial * nk_factorial)

        # Calculate PMF: C(n,k) * p^k * (1-p)^(n-k)
        pmf_value = (binomial_coeff * (self.p ** k) *
                     ((1 - self.p) ** (self.n - k)))

        return pmf_value

    def cdf(self, k):
        """
        Calculates the value of the CDF for a given number of successes

        Args:
            k: number of successes

        Returns:
            CDF value for k
        """
        k = int(k)
        if k < 0:
            return 0

        # Calculate CDF: sum of PMF from 0 to k
        cdf_value = 0
        for i in range(k + 1):
            cdf_value += self.pmf(i)

        return cdf_value
