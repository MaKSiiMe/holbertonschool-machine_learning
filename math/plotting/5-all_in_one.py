#!/usr/bin/env python3
"""Module for plot various graphs in one figure."""
import numpy as np
import matplotlib.pyplot as plt


def all_in_one():
    """Function to plot various graphs in one figure."""

    fig = plt.figure(figsize=(6.4, 4.8), constrained_layout=True)
    gs = fig.add_gridspec(nrows=3, ncols=2)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])
    ax5 = fig.add_subplot(gs[2, :])

    # ---------- Plot 1: Line ----------
    y0 = np.arange(0, 11) ** 3
    ax1.plot(np.arange(0, 11), y0, color='red')
    ax1.set_xlim(0, 10)
    ax1.set_xticks(np.arange(0, 11, 2))
    ax1.set_yticks(np.arange(0, 1001, 500))

    # ---------- Plot 2: Scatter ----------
    mean = [69, 0]
    cov = [[15, 8], [8, 15]]
    np.random.seed(5)
    x1, y1 = np.random.multivariate_normal(mean, cov, 2000).T
    y1 += 180
    ax2.scatter(x1, y1, c='magenta')
    ax2.set_title("Men's Height vs Weight", fontsize='x-small')
    ax2.set_xlabel("Height (in)", fontsize='x-small')
    ax2.set_ylabel("Weight (lbs)", fontsize='x-small')
    ax2.set_xticks(np.arange(60, 81, 10))
    ax2.set_yticks(np.arange(170, 191, 10))

    # ---------- Plot 3: Change scale ----------
    x2 = np.arange(0, 28651, 5730)
    r2 = np.log(0.5)
    t2 = 5730
    y2 = np.exp((r2 / t2) * x2)
    ax3.plot(x2, y2)
    ax3.set_yscale("log")
    ax3.set_xlim(0, 28650)
    ax3.set_title("Exponential Decay of C-14", fontsize='x-small')
    ax3.set_xlabel("Time (years)", fontsize='x-small')
    ax3.set_ylabel("Fraction Remaining", fontsize='x-small')
    ax3.set_xticks(np.arange(0, 30000, 10000))

    # ---------- Plot 4: Two curves ----------
    x3 = np.arange(0, 21000, 1000)
    r3 = np.log(0.5)
    t31 = 5730
    t32 = 1600
    y31 = np.exp((r3 / t31) * x3)
    y32 = np.exp((r3 / t32) * x3)
    ax4.plot(x3, y31, color='red', linestyle='--', label='C-14')
    ax4.plot(x3, y32, color='green', label='Ra-226')
    ax4.set_xlim(0, 20000)
    ax4.set_ylim(0, 1)
    ax4.set_title("Exponential Decay of Radioactive Elements",
                  fontsize='x-small')
    ax4.set_xlabel("Time (years)", fontsize='x-small')
    ax4.set_ylabel("Fraction Remaining", fontsize='x-small')
    ax4.legend(fontsize='x-small', loc='best')
    ax4.set_xticks(np.arange(0, 20001, 5000))
    ax4.set_yticks(np.arange(0, 1.1, 0.5))

    # ---------- Plot 5: Frequency ----------
    np.random.seed(5)
    student_grades = np.random.normal(68, 15, 50)
    bins = np.arange(0, 101, 10)
    ax5.hist(student_grades, bins=bins, edgecolor='black')
    ax5.set_xlim(0, 100)
    ax5.set_ylim(0, 30)
    ax5.set_xticks(bins)
    ax5.set_yticks([0, 10, 20, 30])
    ax5.set_title("Project A", fontsize='x-small')
    ax5.set_xlabel("Grades", fontsize='x-small')
    ax5.set_ylabel("Number of Students", fontsize='x-small')

    fig.suptitle("All in One")

    plt.savefig("all_in_one.png")
    plt.show()
