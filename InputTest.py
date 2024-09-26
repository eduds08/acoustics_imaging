import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from framework import file_m2k as m2k_handler
from framework.data_types import DataInsp
import matplotlib.colors as colors
from scipy.interpolate import interp1d

color = list(colors.TABLEAU_COLORS.values())


class InputTest:
    def __init__(self):
        self.bscan = None
        self.total_time = None
        self.dt = None
        self.microphones_distance = None
        self.microphones_amount = None

        self.time = None

        self.bscan_fmc = None
        self.fmc_emitter = None
        self.gate_start = None

    def load_data_acude(self, file: str, resampled: bool):
        data_acude = loadmat(file)

        if resampled:
            self.bscan = np.load('./acude/bscan_resampled.npy')
            self.time = np.load('./acude/time_resampled.npy')
            self.dt = np.float32(self.time[1] - self.time[0])
        else:
            self.bscan = data_acude['data'].transpose().astype(np.float32)
            rep_rate = data_acude['repRate']
            self.dt = np.float32((1 / rep_rate).item())
            self.time = data_acude['time'][0].astype(np.float32)

        self.total_time = np.int32(len(self.bscan[0, :]))
        self.microphones_amount = np.int32(len(self.bscan[:, 0]))

        self.microphones_distance = np.float32(data_acude['dstep'].item())

    def load_data_panther(self, file_m2k: str):
        data_panther: DataInsp = m2k_handler.read(file_m2k, freq_transd=5, bw_transd=0.5, tp_transd='gaussian', sel_shots=0)

        self.bscan_fmc = data_panther.ascan_data[:, :, :, 0].astype(np.float32)
        self.total_time = np.int32(len(self.bscan_fmc[:, 0, 0]))
        self.dt = np.float32(data_panther.inspection_params.sample_time * 1e-6)
        self.microphones_distance = np.float32(data_panther.probe_params.pitch * 1e-3)
        self.microphones_amount = np.int32(data_panther.probe_params.num_elem)
        self.gate_start = np.float32(data_panther.inspection_params.gate_start)

    def select_fmc_emitter(self, microphone_index):
        self.fmc_emitter = np.int32(microphone_index)

        self.bscan = self.bscan_fmc[:, microphone_index, :].transpose().astype(np.float32)

        padding_zeros = np.int32(self.gate_start / (self.dt * 1e6))
        padding_zeros = np.zeros((self.microphones_amount, padding_zeros))

        self.bscan = np.hstack((padding_zeros, self.bscan), dtype=np.float32)

        self.total_time = np.int32(len(self.bscan[0, :]))

    def select_bscan_interval(self, min_time=None, max_time=None):
        if min_time is None and max_time is not None:
            self.bscan = self.bscan[:, :max_time]
        elif min_time is not None and max_time is None:
            self.bscan = self.bscan[:, min_time:]
        elif min_time is not None and max_time is not None:
            self.bscan = self.bscan[:, min_time:max_time]

        self.total_time = np.int32(len(self.bscan[0, :]))

    def plot_ascans(self):
        plt.figure()

        for mic in range(0, self.microphones_amount, 10):
            plt.plot(self.bscan[mic] + (len(self.bscan[:, 0]) - mic - 1) * 2, label=mic)

        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_bscan(self):
        plt.figure()
        plt.imshow(np.abs(self.bscan), aspect='auto')
        plt.xlabel('Time')
        plt.ylabel('Microphone')
        plt.title(f'B-Scan ({self.microphones_amount}x{self.total_time})')
        plt.grid(True)
        plt.colorbar()
        plt.show()

    def resample_bscan(self, dt_new):
        time_new = np.arange(self.time[0], self.time[-1], dt_new, dtype=np.float32)
        interpolate_f = interp1d(self.time, self.bscan, kind='cubic')
        self.bscan = np.float32(interpolate_f(time_new))

        np.save('./acude/bscan_resampled.npy', self.bscan)
        np.save('./acude/time_resampled.npy', time_new)

    def plot_ascan(self, microphone_index: int):
        plt.figure()
        plt.plot(self.bscan[microphone_index])
        plt.legend()
        plt.grid(True)
        plt.show()
