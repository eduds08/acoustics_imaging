import numpy as np


class SimulationConfig:
    def __init__(self, **simulation_config):
        self.dt = simulation_config['dt']

        self.c = simulation_config['c']

        self.dz = simulation_config['dz']
        self.dx = simulation_config['dx']

        self.grid_size_z = simulation_config['grid_size_z']
        self.grid_size_x = simulation_config['grid_size_x']

        self.grid_size_shape = (self.grid_size_z, self.grid_size_x)

        self.total_time = simulation_config['total_time']

        self.animation_step = simulation_config['animation_step']

        print(f'Total time: {self.total_time}')

        print(f'CFL-Z (np.amax(c)): {np.amax(self.c) * (self.dt / self.dz)}')
        print(f'CFL-X (np.amax(c)): {np.amax(self.c) * (self.dt / self.dx)}')

        print(f'Grid Size (px): ({self.grid_size_z}, {self.grid_size_x})')

        # Pressure fields
        self.p_future = np.zeros(self.grid_size_shape, dtype=np.float32)
        self.p_present = np.zeros(self.grid_size_shape, dtype=np.float32)
        self.p_past = np.zeros(self.grid_size_shape, dtype=np.float32)

        # Partial derivatives
        self.dp_1_z = np.zeros(self.grid_size_shape, dtype=np.float32)
        self.dp_1_x = np.zeros(self.grid_size_shape, dtype=np.float32)
        self.dp_2_z = np.zeros(self.grid_size_shape, dtype=np.float32)
        self.dp_2_x = np.zeros(self.grid_size_shape, dtype=np.float32)

        # CPML
        absorption_layer_size = np.int32(15)
        damping_coefficient = np.float32(3e6)

        x, z = np.meshgrid(np.arange(self.grid_size_x, dtype=np.float32), np.arange(self.grid_size_z, dtype=np.float32))

        # Choose absorbing boundaries
        is_z_absorption = (z > self.grid_size_z - absorption_layer_size) | (z < absorption_layer_size)
        is_x_absorption = (x > self.grid_size_x - absorption_layer_size)

        absorption_coefficient = np.exp(
            -(damping_coefficient * (np.arange(absorption_layer_size) / absorption_layer_size) ** 2) * self.dt
        ).astype(np.float32)

        self.psi_z = np.zeros(self.grid_size_shape, dtype=np.float32)
        self.psi_x = np.zeros(self.grid_size_shape, dtype=np.float32)
        self.phi_z = np.zeros(self.grid_size_shape, dtype=np.float32)
        self.phi_x = np.zeros(self.grid_size_shape, dtype=np.float32)

        self.absorption_z = np.ones(self.grid_size_shape, dtype=np.float32)
        self.absorption_x = np.ones(self.grid_size_shape, dtype=np.float32)

        self.absorption_z[:absorption_layer_size, :] = absorption_coefficient[:, np.newaxis][::-1]
        self.absorption_z[-absorption_layer_size:, :] = absorption_coefficient[:, np.newaxis]
        self.absorption_x[:, :absorption_layer_size] = absorption_coefficient[::-1]
        self.absorption_x[:, -absorption_layer_size:] = absorption_coefficient

        # Goes to GPU
        self.is_z_absorption_int = is_z_absorption.astype(np.int32)
        self.is_x_absorption_int = is_x_absorption.astype(np.int32)
