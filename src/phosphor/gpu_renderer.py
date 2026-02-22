"""wgpu device, pipeline, buffer management, and draw calls."""

import struct

import wgpu

from .constants import BG_COLOR, CURSOR_COLOR, CURSOR_GAP_COLUMNS
from .shader import CURSOR_SHADER, SWEEP_SHADER
from .sweep_buffer import SweepBuffer


class GPURenderer:
    def __init__(self, canvas):
        self.canvas = canvas
        self.adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
        self.device = self.adapter.request_device_sync()
        self.context = canvas.get_context("wgpu")
        self.texture_format = self.context.get_preferred_format(self.adapter)
        self.context.configure(device=self.device, format=self.texture_format)

        self._n_visible = 0
        self._n_columns = 0
        self._buf_version = -1
        self._initialized = False

        # GPU resources (created in setup)
        self.data_buffer = None
        self.channel_params_buffer = None
        self.uniforms_buffer = None
        self.cursor_uniforms_buffer = None
        self.sweep_pipeline = None
        self.cursor_pipeline = None
        self.sweep_bind_group = None
        self.cursor_bind_group = None

    def needs_setup(self, n_visible: int, n_columns: int, buf_version: int) -> bool:
        return (
            not self._initialized
            or n_visible != self._n_visible
            or n_columns != self._n_columns
            or buf_version != self._buf_version
        )

    def setup(self, n_visible: int, n_columns: int, buf_version: int) -> None:
        """Create or recreate all GPU resources for the given dimensions."""
        self._n_visible = n_visible
        self._n_columns = n_columns
        self._buf_version = buf_version

        device = self.device

        # --- Buffers ---
        data_size = n_columns * n_visible * 2 * 4  # float32
        self.data_buffer = device.create_buffer(
            size=max(data_size, 4),
            usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST,
        )

        params_size = n_visible * 8 * 4  # 8 floats per channel
        self.channel_params_buffer = device.create_buffer(
            size=max(params_size, 4),
            usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST,
        )

        self.uniforms_buffer = device.create_buffer(
            size=32,
            usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
        )

        self.cursor_uniforms_buffer = device.create_buffer(
            size=32,
            usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
        )

        # --- Sweep pipeline ---
        sweep_shader = device.create_shader_module(code=SWEEP_SHADER)

        sweep_bgl = device.create_bind_group_layout(
            entries=[
                {
                    "binding": 0,
                    "visibility": wgpu.ShaderStage.VERTEX,
                    "buffer": {
                        "type": wgpu.BufferBindingType.read_only_storage,
                    },
                },
                {
                    "binding": 1,
                    "visibility": wgpu.ShaderStage.VERTEX,
                    "buffer": {
                        "type": wgpu.BufferBindingType.read_only_storage,
                    },
                },
                {
                    "binding": 2,
                    "visibility": wgpu.ShaderStage.VERTEX,
                    "buffer": {
                        "type": wgpu.BufferBindingType.uniform,
                    },
                },
            ]
        )

        sweep_layout = device.create_pipeline_layout(bind_group_layouts=[sweep_bgl])

        self.sweep_pipeline = device.create_render_pipeline(
            layout=sweep_layout,
            vertex={
                "module": sweep_shader,
                "entry_point": "vs_main",
                "buffers": [],
            },
            primitive={
                "topology": wgpu.PrimitiveTopology.line_strip,
            },
            fragment={
                "module": sweep_shader,
                "entry_point": "fs_main",
                "targets": [
                    {
                        "format": self.texture_format,
                        "blend": {
                            "color": {
                                "src_factor": wgpu.BlendFactor.src_alpha,
                                "dst_factor": wgpu.BlendFactor.one_minus_src_alpha,
                                "operation": wgpu.BlendOperation.add,
                            },
                            "alpha": {
                                "src_factor": wgpu.BlendFactor.one,
                                "dst_factor": wgpu.BlendFactor.one_minus_src_alpha,
                                "operation": wgpu.BlendOperation.add,
                            },
                        },
                    }
                ],
            },
        )

        self.sweep_bind_group = device.create_bind_group(
            layout=sweep_bgl,
            entries=[
                {
                    "binding": 0,
                    "resource": {
                        "buffer": self.data_buffer,
                        "offset": 0,
                        "size": self.data_buffer.size,
                    },
                },
                {
                    "binding": 1,
                    "resource": {
                        "buffer": self.channel_params_buffer,
                        "offset": 0,
                        "size": self.channel_params_buffer.size,
                    },
                },
                {
                    "binding": 2,
                    "resource": {
                        "buffer": self.uniforms_buffer,
                        "offset": 0,
                        "size": self.uniforms_buffer.size,
                    },
                },
            ],
        )

        # --- Cursor pipeline ---
        cursor_shader = device.create_shader_module(code=CURSOR_SHADER)

        cursor_bgl = device.create_bind_group_layout(
            entries=[
                {
                    "binding": 0,
                    "visibility": wgpu.ShaderStage.VERTEX | wgpu.ShaderStage.FRAGMENT,
                    "buffer": {
                        "type": wgpu.BufferBindingType.uniform,
                    },
                },
            ]
        )

        cursor_layout = device.create_pipeline_layout(bind_group_layouts=[cursor_bgl])

        self.cursor_pipeline = device.create_render_pipeline(
            layout=cursor_layout,
            vertex={
                "module": cursor_shader,
                "entry_point": "vs_cursor",
                "buffers": [],
            },
            primitive={
                "topology": wgpu.PrimitiveTopology.triangle_list,
            },
            fragment={
                "module": cursor_shader,
                "entry_point": "fs_cursor",
                "targets": [
                    {
                        "format": self.texture_format,
                        "blend": {
                            "color": {
                                "src_factor": wgpu.BlendFactor.src_alpha,
                                "dst_factor": wgpu.BlendFactor.one_minus_src_alpha,
                                "operation": wgpu.BlendOperation.add,
                            },
                            "alpha": {
                                "src_factor": wgpu.BlendFactor.one,
                                "dst_factor": wgpu.BlendFactor.one_minus_src_alpha,
                                "operation": wgpu.BlendOperation.add,
                            },
                        },
                    }
                ],
            },
        )

        self.cursor_bind_group = device.create_bind_group(
            layout=cursor_bgl,
            entries=[
                {
                    "binding": 0,
                    "resource": {
                        "buffer": self.cursor_uniforms_buffer,
                        "offset": 0,
                        "size": self.cursor_uniforms_buffer.size,
                    },
                },
            ],
        )

        self._initialized = True

    def update_and_draw(self, buf: SweepBuffer) -> None:
        """Upload data from SweepBuffer to GPU and render one frame."""
        if self.needs_setup(buf.n_visible, buf.n_columns, buf.version):
            self.setup(buf.n_visible, buf.n_columns, buf.version)
            # Full upload after setup
            data = buf.get_gpu_data()
            self.device.queue.write_buffer(self.data_buffer, 0, data.tobytes())
        else:
            # Incremental upload
            result = buf.get_dirty_gpu_data()
            if result is not None:
                data, col_start, n_cols = result
                byte_offset = col_start * buf.n_visible * 2 * 4
                self.device.queue.write_buffer(self.data_buffer, byte_offset, data.tobytes())

        # Upload channel params
        params = buf.get_channel_params()
        self.device.queue.write_buffer(self.channel_params_buffer, 0, params.tobytes())

        # Upload sweep uniforms
        y_scale = buf.y_scale
        uniforms = struct.pack(
            "<fIIIIIII",
            y_scale,
            buf.n_columns * 2,  # n_display_points
            buf.n_columns,
            buf.sweep_col,
            CURSOR_GAP_COLUMNS,
            buf.n_visible,
            0,  # pad
            0,  # pad
        )
        self.device.queue.write_buffer(self.uniforms_buffer, 0, uniforms)

        # Upload cursor uniforms
        sweep_x = (buf.sweep_col + 0.5) / buf.n_columns * 2.0 - 1.0
        gap_w = CURSOR_GAP_COLUMNS / buf.n_columns * 2.0
        cursor_uniforms = struct.pack(
            "<8f",
            sweep_x,
            sweep_x + gap_w,
            0.0,
            0.0,
            *CURSOR_COLOR,
        )
        self.device.queue.write_buffer(self.cursor_uniforms_buffer, 0, cursor_uniforms)

        # --- Render ---
        try:
            current_texture = self.context.get_current_texture()
        except Exception:
            return

        command_encoder = self.device.create_command_encoder()

        render_pass = command_encoder.begin_render_pass(
            color_attachments=[
                {
                    "view": current_texture.create_view(),
                    "resolve_target": None,
                    "clear_value": BG_COLOR,
                    "load_op": wgpu.LoadOp.clear,
                    "store_op": wgpu.StoreOp.store,
                }
            ],
        )

        # Sweep lines (instanced line-strip)
        render_pass.set_pipeline(self.sweep_pipeline)
        render_pass.set_bind_group(0, self.sweep_bind_group)
        render_pass.draw(self._n_columns * 2, self._n_visible)

        # Cursor overlay quad
        render_pass.set_pipeline(self.cursor_pipeline)
        render_pass.set_bind_group(0, self.cursor_bind_group)
        render_pass.draw(6, 1)

        render_pass.end()
        self.device.queue.submit([command_encoder.finish()])
