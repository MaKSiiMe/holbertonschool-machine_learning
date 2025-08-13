#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt


def bars():
    """Function to plot a stacked bar chart"""
    np.random.seed(5)
    fruit = np.random.randint(0, 20, (4, 3))
    plt.figure(figsize=(6.4, 4.8))

    peoples = ("Farrah", "Fred", "Felicia")
    fruits = ["apples", "bananas", "oranges", "peaches"]
    colors = ['red', 'yellow', '#ff8000', '#ffe5b4']

    bottom = np.zeros(3)
    for i in range(len(fruits)):
        plt.bar(peoples, fruit[i], width=0.5, bottom=bottom,
                color=colors[i], label=fruits[i])
        bottom += fruit[i]

    plt.ylabel("Quantity of Fruit")
    plt.yticks(np.arange(0, 81, 10))
    plt.title("Number of Fruit per Person")
    plt.legend()
    plt.ylim(0, 80)
    plt.show()
