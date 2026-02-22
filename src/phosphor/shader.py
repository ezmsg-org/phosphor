"""WGSL shader source strings for sweep lines and cursor overlay."""

SWEEP_SHADER = """
struct Uniforms {
    y_scale: f32,
    n_display_points: u32,
    n_columns: u32,
    sweep_col: u32,
    cursor_gap: u32,
    n_visible: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<storage, read> data: array<f32>;
@group(0) @binding(1) var<storage, read> channel_params: array<f32>;
@group(0) @binding(2) var<uniform> uniforms: Uniforms;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec4<f32>,
}

@vertex
fn vs_main(
    @builtin(vertex_index) vertex_index: u32,
    @builtin(instance_index) instance_index: u32,
) -> VertexOutput {
    var out: VertexOutput;

    let column = vertex_index / 2u;
    let sub = vertex_index % 2u;

    // Read display data value (column-major interleaved min/max)
    let data_index = (column * uniforms.n_visible + instance_index) * 2u + sub;
    let value = data[data_index];

    // Read per-channel params (8 floats per channel)
    let param_base = instance_index * 8u;
    let y_offset = channel_params[param_base + 0u];
    let color = vec4<f32>(
        channel_params[param_base + 4u],
        channel_params[param_base + 5u],
        channel_params[param_base + 6u],
        channel_params[param_base + 7u],
    );

    let x = (f32(column) + 0.5) / f32(uniforms.n_columns) * 2.0 - 1.0;
    let y = value * uniforms.y_scale + y_offset;

    out.position = vec4<f32>(x, y, 0.0, 1.0);
    out.color = color;

    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return in.color;
}
"""

CURSOR_SHADER = """
struct CursorUniforms {
    x_left: f32,
    x_right: f32,
    _pad0: f32,
    _pad1: f32,
    color: vec4<f32>,
}

@group(0) @binding(0) var<uniform> cursor: CursorUniforms;

@vertex
fn vs_cursor(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4<f32> {
    // Full-height quad from two triangles
    var px: array<f32, 6> = array<f32, 6>(0.0, 1.0, 0.0, 1.0, 1.0, 0.0);
    var py: array<f32, 6> = array<f32, 6>(-1.0, -1.0, 1.0, -1.0, 1.0, 1.0);

    let x = mix(cursor.x_left, cursor.x_right, px[vi]);
    let y = py[vi];

    return vec4<f32>(x, y, 0.0, 1.0);
}

@fragment
fn fs_cursor() -> @location(0) vec4<f32> {
    return cursor.color;
}
"""
