import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat


class InputTest:
    def __init__(self):
        self.bscan = None
        self.dt = None
        self.gate_start = None
        self.microphones_distance = None
        self.total_time = None
        self.microphones_amount = None

    def load_data_acude(self, file: str):
        data_acude = loadmat(file)

        self.bscan = data_acude['data'].transpose().astype(np.float32)

        self.total_time = np.int32(len(self.bscan[0, :]))
        self.microphones_amount = np.int32(len(self.bscan[:, 0]))

        rep_rate = data_acude['repRate']
        self.dt = np.float32((1 / rep_rate).item())

        self.gate_start = np.float32(0)

        self.microphones_distance = np.float32(data_acude['dstep'])

    def load_data_panther(self, folder: str):
        pass

        # Add zeros, according to gate_start, to the beginning of array
        gate_start_value = np.float32(0)
        if selected_test[0] != './panther/teste7_results':
            with open(f'{selected_test[0]}/inspection_params.txt', 'r') as f:
                for line in f:
                    if line.startswith('gate_start'):
                        gate_start_value = np.float32(line.split('=')[1].strip())
                        break
        padding_zeros = np.int32(gate_start_value / sample_time)
        padding_zeros = np.zeros((len(recorded_pressure_bscan[:, 0]), padding_zeros))
        recorded_pressure_bscan = np.hstack((padding_zeros, recorded_pressure_bscan), dtype=np.float32)

    def select_bscan_interval(self, min_time=None, max_time=None):
        if min_time is None and max_time is not None:
            self.bscan = self.bscan[:, :max_time]
        elif min_time is not None and max_time is None:
            self.bscan = self.bscan[:, min_time:]
        elif min_time is not None and max_time is not None:
            self.bscan = self.bscan[:, min_time:max_time]

        self.total_time = np.int32(len(self.bscan[0, :]))

    def plot_bscan(self):
        plt.figure()
        plt.imshow(np.abs(self.bscan), aspect='auto')
        plt.xlabel('Time')
        plt.ylabel('Microphone')
        plt.title(f'B-Scan ({self.microphones_amount}x{self.total_time})')
        plt.grid()
        plt.colorbar()
        plt.show()
