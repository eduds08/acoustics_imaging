import numpy as np
import os
import re
from InputTest import InputTest
import pyqtgraph as pg
from pyqtgraph.widgets.RawImageWidget import RawImageGLWidget
from SimulationConfig import SimulationConfig
from WebGpuHandler import WebGpuHandler
import matplotlib.pyplot as plt
import matplotlib


class ReverseTimeMigration(SimulationConfig):
    def __init__(self, **simulation_config):
        super().__init__(**simulation_config)

        x = np.linspace(0, self.total_time, self.total_time)
        sigma = 10
        mu = 50
        amplitude = 1
        self.source = amplitude * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))
        self.source[x < mu - 3 * sigma] = 0