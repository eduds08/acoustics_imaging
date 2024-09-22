import numpy as np
from SyntheticAcouSim import SyntheticAcouSim
from SyntheticTimeReversal import SyntheticTimeReversal
from SyntheticReverseTimeMigration import SyntheticReverseTimeMigration
from functions import convert_image_to_matrix

c, microphone_z, microphone_x = convert_image_to_matrix('./map.png')

grid_size_z = np.int32(len(c[:, 0]))
grid_size_x = np.int32(len(c[0, :]))
grid_size_shape = (grid_size_z, grid_size_x)

dt = np.float32(1e-3)

# Spatial steps (m/px)
dz = np.float32(np.amax(c) * dt / 0.5)
dx = np.float32(np.amax(c) * dt / 0.5)

simulation_config = {
    'dt': dt,
    'c': c,
    'dz': dz,
    'dx': dx,
    'grid_size_z': grid_size_z,
    'grid_size_x': grid_size_x,
    'total_time': np.int32(1000),
    'medium_c': np.float32(1500),
}

synthetic_config = {
    'source_z': microphone_z[0],
    'source_x': microphone_x[0],
    'microphones_amount': np.int32(len(microphone_z)),
    'microphone_z': microphone_z,
    'microphone_x': microphone_x,
}

simulation_config.update(synthetic_config)

acou_sim = SyntheticAcouSim(**simulation_config)
acou_sim.run(generate_video=True, animation_step=15)

tr_sim = SyntheticTimeReversal(**simulation_config)
tr_sim.run(generate_video=True, animation_step=15)

rtm_sim = SyntheticReverseTimeMigration(**simulation_config)
rtm_sim.run(generate_video=True, animation_step=15)
