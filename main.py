import numpy as np
from ReverseTimeMigration import ReverseTimeMigration
from TimeReversal import TimeReversal
from InputTest import InputTest
import matplotlib.pyplot as plt
from functions import plot_l2_norm

dados = 'panther'
# dados = 'panther'

input_test = InputTest()

if dados == 'acude':
    input_test.load_data_acude(file='./acude/azulPerpendicular1_Variables.mat')
    # input_test.plot_bscan()
    input_test.select_bscan_interval(min_time=None, max_time=45000)
    # input_test.plot_bscan()
elif dados == 'panther':
    input_test.load_data_panther(file_m2k='./arquivos_m2k/teste2_perto_fio_a_esquerda.m2k')

size_meters_z = np.float32(300)
size_meters_x = np.float32(300)  # Não deixar abaixo de 40e-3 para os testes do panther. Os microfones ocupam 38.4e-3 metros.
dz = np.float32(3e-1)
dx = np.float32(3e-1)

grid_size_z = np.int32(size_meters_z / dz)
grid_size_x = np.int32(size_meters_x / dx)

grid_size_shape = (grid_size_z, grid_size_x)

c = np.float32(1500)
c = np.full(grid_size_shape, c, dtype=np.float32)

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

# Plota a L2-Norm do último Time Reversal simulado
# plot_l2_norm()

if dados == 'acude':
    tr_sim = TimeReversal(**simulation_config)
    tr_sim.run(real_time_animation=False)
elif dados == 'panther':
    for microphone_index in range(input_test.microphones_amount):
        input_test.select_fmc_emitter(microphone_index=microphone_index)

        print(f'Microfone {microphone_index}/63')

        tr_sim = TimeReversal(**simulation_config)
        tr_sim.run(real_time_animation=False)

        rtm_sim = ReverseTimeMigration(**simulation_config)
        rtm_sim.run(real_time_animation=False)
