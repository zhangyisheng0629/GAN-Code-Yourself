#!/usr/bin/python
# author eson
import matplotlib.pyplot as plt
class LossGraph:
    def __init__(self):
        self.data=[]

    def add(self,loss):
        self.data.append(loss)

    def draw(self):
        plt.plot(self.data)
        plt.show()
