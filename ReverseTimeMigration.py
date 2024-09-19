import numpy as np
from InputTest import InputTest
from SimulationConfig import SimulationConfig
from WebGpuHandler import WebGpuHandler
import pyqtgraph as pg
from pyqtgraph.widgets.RawImageWidget import RawImageGLWidget
import matplotlib.pyplot as plt
import matplotlib


class ReverseTimeMigration(SimulationConfig):
    def __init__(self, **simulation_config):
        super().__init__(**simulation_config)

        input_test: InputTest = simulation_config['input_test']

        self.total_time = input_test.total_time
        print(f'Total time: {self.total_time}')

        self.emitter_index = input_test.fmc_emitter

        x = np.linspace(0, self.total_time, self.total_time)
        sigma = 10
        mu = 50
        amplitude = 1
        self.source = amplitude * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))
        self.source[x < mu - 3 * sigma] = 0

        self.source = np.float32(self.source)

        self.source_z = np.int32(np.load('./TimeReversal/emitter_z.npy'))
        self.source_x = np.int32(np.load('./TimeReversal/emitter_x.npy'))

        # Up-going pressure fields (Flipped Time Reversal)
        self.p_future_flipped_tr = np.zeros(self.grid_size_shape, dtype=np.float32)
        self.p_present_flipped_tr = np.load(f'./TimeReversal/frame_{self.total_time - 2}.npy')
        self.p_past_flipped_tr = np.load(f'./TimeReversal/frame_{self.total_time - 1}.npy')

        # Partial derivatives
        self.dp_1_z_flipped_tr = np.zeros(self.grid_size_shape, dtype=np.float32)
        self.dp_1_x_flipped_tr = np.zeros(self.grid_size_shape, dtype=np.float32)
        self.dp_2_z_flipped_tr = np.zeros(self.grid_size_shape, dtype=np.float32)
        self.dp_2_x_flipped_tr = np.zeros(self.grid_size_shape, dtype=np.float32)

        # CPML (Flipped Time Reversal)
        self.psi_z_flipped_tr = self.psi_z.copy()
        self.psi_x_flipped_tr = self.psi_x.copy()
        self.phi_z_flipped_tr = self.phi_z.copy()
        self.phi_x_flipped_tr = self.phi_x.copy()
        self.absorption_z_flipped_tr = self.absorption_z.copy()
        self.absorption_x_flipped_tr = self.absorption_x.copy()
        self.is_z_absorption_int_flipped_tr = self.is_z_absorption_int.copy()
        self.is_x_absorption_int_flipped_tr = self.is_x_absorption_int.copy()

        self.info_i32 = np.array(
            [
                self.grid_size_z,
                self.grid_size_x,
                self.source_z,
                self.source_x,
                0,
            ],
            dtype=np.int32
        )

        self.info_f32 = np.array(
            [
                self.dz,
                self.dx,
                self.dt,
            ],
            dtype=np.float32
        )

        self.wgpu_handler = None
        self.config_gpu()

    def config_gpu(self):
        self.wgpu_handler = WebGpuHandler(self.grid_size_z, self.grid_size_x)

        shader_file = open('./reverse_time_migration.wgsl')
        shader_string = (shader_file.read()
                         .replace('wsz', f'{self.wgpu_handler.ws[0]}')
                         .replace('wsx', f'{self.wgpu_handler.ws[1]}'))
        shader_file.close()

        self.wgpu_handler.shader_module = self.wgpu_handler.device.create_shader_module(code=shader_string)

        wgsl_data = {
            'infoI32': self.info_i32,
            'infoF32': self.info_f32,
            'source': self.source,
            'c': self.c,
            'p_future': self.p_future,
            'p_present': self.p_present,
            'p_past': self.p_past,
            'dp_1_z': self.dp_1_z,
            'dp_1_x': self.dp_1_x,
            'dp_2_z': self.dp_2_z,
            'dp_2_x': self.dp_2_x,
            'psi_z': self.psi_z,
            'psi_x': self.psi_x,
            'phi_z': self.phi_z,
            'phi_x': self.phi_x,
            'absorption_z': self.absorption_z,
            'absorption_x': self.absorption_x,
            'is_z_absorption': self.is_z_absorption_int,
            'is_x_absorption': self.is_x_absorption_int,
            'p_future_flipped_tr': self.p_future_flipped_tr,
            'p_present_flipped_tr': self.p_present_flipped_tr,
            'p_past_flipped_tr': self.p_past_flipped_tr,
            'dp_1_z_flipped_tr': self.dp_1_z_flipped_tr,
            'dp_1_x_flipped_tr': self.dp_1_x_flipped_tr,
            'dp_2_z_flipped_tr': self.dp_2_z_flipped_tr,
            'dp_2_x_flipped_tr': self.dp_2_x_flipped_tr,
            'psi_z_flipped_tr': self.psi_z_flipped_tr,
            'psi_x_flipped_tr': self.psi_x_flipped_tr,
            'phi_z_flipped_tr': self.phi_z_flipped_tr,
            'phi_x_flipped_tr': self.phi_x_flipped_tr,
            'absorption_z_flipped_tr': self.absorption_z_flipped_tr,
            'absorption_x_flipped_tr': self.absorption_x_flipped_tr,
            'is_z_absorption_flipped_tr': self.is_z_absorption_int_flipped_tr,
            'is_x_absorption_flipped_tr': self.is_x_absorption_int_flipped_tr,
        }

        shader_lines = list(shader_string.split('\n'))
        self.wgpu_handler.create_buffers(wgsl_data, shader_lines)

    def run(self, real_time_animation: bool):
        compute_forward_diff = self.wgpu_handler.create_compute_pipeline("forward_diff")
        compute_after_forward = self.wgpu_handler.create_compute_pipeline("after_forward")
        compute_backward_diff = self.wgpu_handler.create_compute_pipeline("backward_diff")
        compute_after_backward = self.wgpu_handler.create_compute_pipeline("after_backward")
        compute_sim_flipped_tr = self.wgpu_handler.create_compute_pipeline("sim_flipped_tr")
        compute_sim = self.wgpu_handler.create_compute_pipeline("sim")
        compute_incr_time = self.wgpu_handler.create_compute_pipeline("incr_time")

        # GUI (animação)
        if real_time_animation:
            vminmax = 1
            vscale = 1
            surface_format = pg.QtGui.QSurfaceFormat()
            surface_format.setSwapInterval(0)
            pg.QtGui.QSurfaceFormat.setDefaultFormat(surface_format)
            app = pg.QtWidgets.QApplication([])
            raw_image_widget = RawImageGLWidget()
            raw_image_widget.setWindowFlags(pg.QtCore.Qt.WindowType.FramelessWindowHint)
            raw_image_widget.resize(vscale * self.grid_size_x, vscale * self.grid_size_z)
            raw_image_widget.show()
            colormap = plt.get_cmap("bwr")
            norm = matplotlib.colors.Normalize(vmin=-vminmax, vmax=vminmax)

        accumulated_product = np.zeros_like(self.p_future)

        for i in range(self.total_time):
            command_encoder = self.wgpu_handler.device.create_command_encoder()
            compute_pass = command_encoder.begin_compute_pass()

            for index, bind_group in enumerate(self.wgpu_handler.bind_groups):
                compute_pass.set_bind_group(index, bind_group, [], 0, 999999)

            compute_pass.set_pipeline(compute_forward_diff)
            compute_pass.dispatch_workgroups(self.grid_size_z // self.wgpu_handler.ws[0],
                                             self.grid_size_x // self.wgpu_handler.ws[1])

            compute_pass.set_pipeline(compute_after_forward)
            compute_pass.dispatch_workgroups(self.grid_size_z // self.wgpu_handler.ws[0],
                                             self.grid_size_x // self.wgpu_handler.ws[1])

            compute_pass.set_pipeline(compute_backward_diff)
            compute_pass.dispatch_workgroups(self.grid_size_z // self.wgpu_handler.ws[0],
                                             self.grid_size_x // self.wgpu_handler.ws[1])

            compute_pass.set_pipeline(compute_after_backward)
            compute_pass.dispatch_workgroups(self.grid_size_z // self.wgpu_handler.ws[0],
                                             self.grid_size_x // self.wgpu_handler.ws[1])

            compute_pass.set_pipeline(compute_sim_flipped_tr)
            compute_pass.dispatch_workgroups(self.grid_size_z // self.wgpu_handler.ws[0],
                                             self.grid_size_x // self.wgpu_handler.ws[1])

            compute_pass.set_pipeline(compute_sim)
            compute_pass.dispatch_workgroups(self.grid_size_z // self.wgpu_handler.ws[0],
                                             self.grid_size_x // self.wgpu_handler.ws[1])

            compute_pass.set_pipeline(compute_incr_time)
            compute_pass.dispatch_workgroups(1)

            compute_pass.end()
            self.wgpu_handler.device.queue.submit([command_encoder.finish()])

            """ READ BUFFERS """
            self.p_future = (np.asarray(self.wgpu_handler.device.queue.read_buffer(self.wgpu_handler.buffers['b4']).cast("f"))
                             .reshape(self.grid_size_shape))
            self.p_future_flipped_tr = (np.asarray(self.wgpu_handler.device.queue.read_buffer(self.wgpu_handler.buffers['b19']).cast("f"))
                             .reshape(self.grid_size_shape))

            # Atualiza a GUI
            if real_time_animation:
                if not i % self.animation_step:
                    raw_image_widget.setImage(colormap(norm(self.p_future.T)), levels=[0, 1])
                    app.processEvents()
                    plt.pause(1e-12)

            current_product = self.p_future * self.p_future_flipped_tr
            accumulated_product += current_product

            # Save last frame as .npy
            if i == self.total_time - 1:
                np.save(f'./ReverseTimeMigration/frame_{self.emitter_index}.npy', accumulated_product)

            if i % 100 == 0:
                print(f'Reverse Time Migration - i={i}')

        print('Reverse Time Migration finished.')
