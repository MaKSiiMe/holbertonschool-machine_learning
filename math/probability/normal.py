#!/usr/bin/env python3
"""Module for Normal distribution"""


class Normal:
    """Represents a Normal distribution"""

    def __init__(self, data=None, mean=0., stddev=1.):
        """
        Initialize Normal distribution

        Args:
            data: list of data to estimate the distribution
            mean: mean of the distribution
            stddev: standard deviation of the distribution
        """
        if data is None:
            if stddev <= 0:
                raise ValueError("stddev must be a positive value")
            self.mean = float(mean)
            self.stddev = float(stddev)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")

            # Calculate mean
            self.mean = float(sum(data) / len(data))

            # Calculate standard deviation
            variance = sum((x - self.mean) ** 2 for x in data) / len(data)
            self.stddev = float(variance ** 0.5)

    def z_score(self, x):
        """
        Calculates the z-score of a given x-value

        Args:
            x: x-value

        Returns:
            z-score of x
        """
        return (x - self.mean) / self.stddev

    def x_value(self, z):
        """
        Calculates the x-value of a given z-score

        Args:
            z: z-score

        Returns:
            x-value of z
        """
        return self.mean + z * self.stddev

    def pdf(self, x):
        """
        Calculates the value of the PDF for a given x-value

        Args:
            x: x-value

        Returns:
            PDF value for x
        """
        pi = 3.1415926536
        e = 2.7182818285

        # Calculate PDF: (1 / (σ√(2π))) * e^(-((x-μ)²) / (2σ²))
        coefficient = 1 / (self.stddev * ((2 * pi) ** 0.5))
        exponent = -(((x - self.mean) ** 2) / (2 * (self.stddev ** 2)))
        pdf_value = coefficient * (e ** exponent)

        return pdf_value

    def cdf(self, x):
        """
        Calculates the value of the CDF for a given x-value

        Args:
            x: x-value

        Returns:
            CDF value for x
        """
        pi = 3.1415926536

        # Calculate z-score
        z = (x - self.mean) / (self.stddev * (2 ** 0.5))

        # Approximate erf using Taylor series
        erf = (2 / (pi ** 0.5)) * (z - (z ** 3) / 3 + (z ** 5) / 10 -
                                       (z ** 7) / 42 + (z ** 9) / 216)

        # Calculate CDF: 0.5 * (1 + erf((x - μ) / (σ√2)))
        cdf_value = 0.5 * (1 + erf)

        return cdf_value
