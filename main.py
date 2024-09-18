import numpy as np
from TimeReversal import TimeReversal
from InputTest import InputTest
import matplotlib.pyplot as plt

input_test = InputTest()
input_test.load_data_acude(file='./acude/azulPerpendicular1_Variables.mat')

size_meters_z = np.float32(250)
size_meters_x = np.float32(250)
dz = np.float32(3e-1)
dx = np.float32(3e-1)

grid_size_z = np.int32(size_meters_z / dz)
grid_size_x = np.int32(size_meters_x / dx)

grid_size_shape = (grid_size_z, grid_size_x)

c = np.float32(1500)
c = np.full(grid_size_shape, c, dtype=np.float32)

# input_test.plot_bscan()

input_test.select_bscan_interval(min_time=None, max_time=45000)

# input_test.plot_bscan()

# Add zeros, according to gate_start, to the beginning of array
# gate_start_value = np.float32(0)
# padding_zeros = np.int32(gate_start_value / sample_time)
# padding_zeros = np.zeros((len(recorded_pressure_bscan[:, 0]), padding_zeros))

# recorded_pressure_bscan = np.hstack((padding_zeros, recorded_pressure_bscan), dtype=np.float32)
#
# recorded_pressure_bscan = recorded_pressure_bscan[:, :50000]
# plot_imshow(np.abs(recorded_pressure_bscan), 'Raw BScan', {}, aspect='auto')



simulation_config = {
    'dt': input_test.dt,
    'c': c,
    'dz': dz,
    'dx': dx,
    'grid_size_z': grid_size_z,
    'grid_size_x': grid_size_x,
    'total_time': input_test.total_time,
    'animation_step': np.int32(100),
}

tr_config = {
    'input_test': input_test,
    'padding_zeros': np.int32(0),
}

simulation_config.update(tr_config)

l2 = np.load('./TimeReversal/l2_norm.npy')

# plt.figure()
# plt.imshow(l2, aspect='auto')
# plt.title(f'L2-Norm')
# plt.grid()
# plt.colorbar()
# plt.show()

tr_sim = TimeReversal(**simulation_config)
tr_sim.run(create_animation=False, cmap='bwr')
