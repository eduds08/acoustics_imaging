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


class TimeReversal(SimulationConfig):
    def __init__(self, **simulation_config):
        super().__init__(**simulation_config)

        self.wgpu_handler = None

        input_test: InputTest = simulation_config['input_test']

        self.bscan = input_test.bscan
        self.microphones_distance = input_test.microphones_distance
        self.microphones_amount = input_test.microphones_amount

        # self.setup_folders()

        self.microphone_z = []
        for rp in range(self.microphones_amount):
            self.microphone_z.append((self.microphones_distance * rp) / self.dz)
        self.microphone_z = (np.int32(np.asarray(self.microphone_z))
                           + np.int32((self.grid_size_z - self.microphone_z[-1]) / 2))
        self.microphone_x = np.full(self.microphones_amount, 0, dtype=np.int32)

        self.flipped_bscan = self.bscan[:, ::-1].astype(np.float32)

        plt.figure()
        plt.imshow(np.abs(self.flipped_bscan), aspect='auto')
        plt.xlabel('Time')
        plt.ylabel('Microphone')
        plt.title(f'Flipped B-Scan ({self.microphones_amount}x{self.total_time})')
        plt.grid()
        plt.colorbar()
        plt.show()

        # self.padding_zeros = simulation_config['padding_zeros']
        # padded_recorded_pressure = np.pad(flipped_recorded_pressure, (0, self.padding_zeros), mode='constant')

        self.info_i32 = np.array(
            [
                self.grid_size_z,
                self.grid_size_x,
                self.microphones_amount,
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

        self.config_gpu()

    def config_gpu(self):
        self.wgpu_handler = WebGpuHandler(self.grid_size_z, self.grid_size_x)

        shader_file = open('./time_reversal.wgsl')
        shader_string = (shader_file.read()
                         .replace('wsz', f'{self.wgpu_handler.ws[0]}')
                         .replace('wsx', f'{self.wgpu_handler.ws[1]}'))

        matches = re.findall(r'@binding\((\d+)\)', shader_string)
        last_binding = int(matches[-1])

        aux_string = ''
        for i in range(self.microphones_amount):
            aux_string += f'''@group(0) @binding({i + (last_binding + 1)})
            var<storage,read> flipped_microphone_{i}: array<f32>;\n\n'''

        shader_string = shader_string.replace('//FLIPPED_MICROPHONES_BINDINGS', aux_string)

        aux_string = ''
        for i in range(self.microphones_amount):
            aux_string += f'''if (microphone_index == {i})
                    {{
                        p_future[zx(z, x)] += flipped_microphone_{i}[infoI32.i];
                    }}\n'''

        shader_string = shader_string.replace('//FLIPPED_MICROPHONES_SIM', aux_string)

        shader_file.close()

        self.wgpu_handler.shader_module = self.wgpu_handler.device.create_shader_module(code=shader_string)

        wgsl_data = {
            'infoI32': self.info_i32,
            'infoF32': self.info_f32,
            'c': self.c,
            'microphone_z': self.microphone_z,
            'microphone_x': self.microphone_x,
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
        }

        for i in range(self.microphones_amount):
            wgsl_data[f'flipped_microphone_{i}'] = np.ascontiguousarray(self.flipped_bscan[i])

        shader_lines = list(shader_string.split('\n'))
        self.wgpu_handler.create_buffers(wgsl_data, shader_lines)

    def run(self, create_animation: bool, **plt_kwargs):
        compute_forward_diff = self.wgpu_handler.create_compute_pipeline("forward_diff")
        compute_after_forward = self.wgpu_handler.create_compute_pipeline("after_forward")
        compute_backward_diff = self.wgpu_handler.create_compute_pipeline("backward_diff")
        compute_after_backward = self.wgpu_handler.create_compute_pipeline("after_backward")
        compute_sim = self.wgpu_handler.create_compute_pipeline("sim")
        compute_incr_time = self.wgpu_handler.create_compute_pipeline("incr_time")

        # if create_animation:
        #     clear_folder(self.animation_folder)

        # scatter_kwargs = {
        #     'microphones_amount': self.microphones_amount,
        #     'microphone_z': self.microphone_z,
        #     'microphone_x': self.microphone_x,
        # }
        #
        # l2_norm = np.zeros_like(self.p_future)

        # GUI (animação)
        # vminmax = 1e3
        # vscale = 1
        # surface_format = pg.QtGui.QSurfaceFormat()
        # surface_format.setSwapInterval(0)
        # pg.QtGui.QSurfaceFormat.setDefaultFormat(surface_format)
        # app = pg.QtWidgets.QApplication([])
        # raw_image_widget = RawImageGLWidget()
        # raw_image_widget.setWindowFlags(pg.QtCore.Qt.WindowType.FramelessWindowHint)
        # raw_image_widget.resize(vscale * self.grid_size_x, vscale * self.grid_size_z)
        # raw_image_widget.show()
        # colormap = plt.get_cmap("bwr")
        # norm = matplotlib.colors.Normalize(vmin=-vminmax, vmax=vminmax)

        l2_norm = np.zeros(self.grid_size_shape, dtype=np.float32)

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

            compute_pass.set_pipeline(compute_sim)
            compute_pass.dispatch_workgroups(self.grid_size_z // self.wgpu_handler.ws[0],
                                             self.grid_size_x // self.wgpu_handler.ws[1])

            compute_pass.set_pipeline(compute_incr_time)
            compute_pass.dispatch_workgroups(1)

            compute_pass.end()
            self.wgpu_handler.device.queue.submit([command_encoder.finish()])

            """ READ BUFFERS """
            self.p_future = (np.asarray(self.wgpu_handler.device.queue.read_buffer(self.wgpu_handler.buffers['b5']).cast("f"))
                             .reshape(self.grid_size_shape))

            # Atualiza a GUI
            # if not i % self.animation_step:
            #     raw_image_widget.setImage(colormap(norm(self.p_future.T)), levels=[0, 1])
            #     app.processEvents()
            #     plt.pause(1e-12)

            l2_norm += np.square(self.p_future)
            #
            # if i % self.animation_step == 0:
            #     if create_animation:
            #         save_imshow(
            #             data=self.p_future,
            #             title=f'Time Reversal',
            #             path=f'{self.animation_folder}/plot_{i}.png',
            #             scatter_kwargs=scatter_kwargs,
            #             **plt_kwargs,
            #         )
            #
            if i % 100 == 0:
                print(f'Time Reversal - i={i}')

        print('Time Reversal finished.')

        l2_norm = np.sqrt(l2_norm)

        np.save(f'./TimeReversal/l2_norm.npy', l2_norm)

        # if create_animation:
        #     create_ffmpeg_animation(self.animation_folder, 'a.mp4', self.tr_total_time, self.animation_step)

    # def setup_folders(self):
    #     self.folder = './TimeReversal'
    #     self.animation_folder = f'{self.folder}/animation'
    #
    #     os.makedirs(self.folder, exist_ok=True)
    #     os.makedirs(self.animation_folder, exist_ok=True)
    #
    #     clear_folder(self.folder)
