#!/usr/bin/env python3
"""Module for Exponential distribution"""


class Exponential:
    """Represents an Exponential distribution"""

    def __init__(self, data=None, lambtha=1.):
        """
        Initialize Exponential distribution

        Args:
            data: list of data to estimate the distribution
            lambtha: expected number of occurences in a given time frame
        """
        if data is None:
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
            self.lambtha = float(lambtha)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            # For exponential distribution, lambtha = 1 / mean
            self.lambtha = 1 / float(sum(data) / len(data))

    def pdf(self, x):
        """
        Calculates the value of the PDF for a given time period

        Args:
            x: time period

        Returns:
            PDF value for x
        """
        if x < 0:
            return 0

        # Calculate PDF: λ * e^(-λx)
        e = 2.7182818285
        pdf_value = self.lambtha * (e ** (-self.lambtha * x))
        return pdf_value

    def cdf(self, x):
        """
        Calculates the value of the CDF for a given time period

        Args:
            x: time period

        Returns:
            CDF value for x
        """
        if x < 0:
            return 0

        # Calculate CDF: 1 - e^(-λx)
        e = 2.7182818285
        cdf_value = 1 - (e ** (-self.lambtha * x))
        return cdf_value
