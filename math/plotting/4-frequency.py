#!/usr/bin/env python3
"""Module for plot the frequency of student grades."""
import numpy as np
import matplotlib.pyplot as plt


def frequency():
    """Function to plot the frequency of student grades."""
    np.random.seed(5)
    student_grades = np.random.normal(68, 15, 50)
    plt.figure(figsize=(6.4, 4.8))

    plt.xlabel("Grades")
    plt.ylabel("Number of Students")
    plt.title("Project A")
    plt.xlim(0, 100)
    plt.ylim(0, 30)
    plt.hist(student_grades, bins=np.arange(0, 101, 10), edgecolor='black')
    plt.xticks(np.arange(0, 101, 10))
    plt.savefig("Frequency.png")
    plt.show()
