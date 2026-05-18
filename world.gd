extends Node

enum PresetId { BASELINE, FIRST }
@export var preset_id : PresetId = PresetId.BASELINE

var rd : RenderingDevice

var grid_width : int   = 1024
var grid_height : int   = 1024
var steps_per_frame : int   = 1

var mu_r : float = 0.155
var mu_g : float = 0.155
var mu_b : float = 0.155

var sigma_r : float = 0.127
var sigma_g : float = 0.127
var sigma_b : float = 0.127

var dt : float = 0.1

var kernel_radius_r : float = 10.0
var kernel_radius_g : float = 10.0
var kernel_radius_b : float = 10.0

var coeff_rr : float = 0.5
var coeff_rg : float = 0.5
var coeff_rb : float = 0.5
var coeff_gr : float = 0.5
var coeff_gg : float = 0.5
var coeff_gb : float = 0.5
var coeff_br : float = 0.5
var coeff_bg : float = 0.5
var coeff_bb : float = 0.5

var seed_r: float = 0.1
var seed_g: float = 0.1
var seed_b: float = 0.1

var tex_a      : RID
var tex_b      : RID
var tex_a_view : RID
var tex_b_view : RID
var current_is_a : bool = true

var sim_shader    : RID
var sim_pipeline  : RID
var brush_shader  : RID
var brush_pipeline: RID

var sim_set_ab : RID = RID()
var sim_set_ba : RID = RID()
var brush_set_a : RID = RID()
var brush_set_b : RID = RID()

var sampler_rid : RID

var brush_active : bool     = false
var brush_pos    : Vector2i = Vector2i.ZERO
var brush_radius : int      = 10
var brush_color  : Color    = Color(1, 1, 1, 1.0)

@onready var output_rect: TextureRect = $TextureRect
var display_tex := Texture2DRD.new()


func _ready() -> void:
	randomize()
	rd = RenderingServer.get_rendering_device()

	_apply_preset(preset_id)
	_create_textures()
	_create_sampler()
	_compile_shaders()
	_create_uniform_sets()
	_seed_random()

	display_tex.texture_rd_rid = current_tex_rid()
	output_rect.texture = display_tex


func _apply_preset(id: PresetId) -> void:
	match id:
		PresetId.BASELINE:
			grid_width = 256
			grid_height = 256
			steps_per_frame = 4

			mu_r = 0.068
			mu_g = 0.118
			mu_b = 0.205

			sigma_r = 0.014
			sigma_g = 0.038
			sigma_b = 0.082

			dt = 0.0016

			kernel_radius_r = 3.0
			kernel_radius_g = 7.0
			kernel_radius_b = 18.0

			brush_radius = 10
			brush_color = Color(1,0,0,1)

			seed_r = 0.0
			seed_g = 0.0
			seed_b = 0.0


			coeff_rr = 0.14
			coeff_rg = -0.06
			coeff_rb = -0.08

			coeff_gr = 0.18
			coeff_gg = 0.10
			coeff_gb = -0.04

			coeff_br = -0.02
			coeff_bg = 0.16
			coeff_bb = -0.10
	
	
		PresetId.FIRST:
			grid_width = 256
			grid_height = 256
			steps_per_frame = 4

			mu_r = 0.11     # membrane
			mu_g = 0.18     # metabolism
			mu_b = 0.29     # signaling

			sigma_r = 0.030
			sigma_g = 0.060
			sigma_b = 0.110

			dt = 0.0038

			kernel_radius_r = 5.0
			kernel_radius_g = 10.0
			kernel_radius_b = 24.0

			brush_radius = 10
			brush_color = Color(1,0,0,1)

			seed_r = 0.006
			seed_g = 0.010
			seed_b = 0.004

			coeff_rr = 0.34
			coeff_rg = -0.10
			coeff_rb = -0.08

			coeff_gr = 0.26
			coeff_gg = 0.14
			coeff_gb = -0.03

			coeff_br = -0.18
			coeff_bg = 0.28
			coeff_bb = 0.02


func _create_textures() -> void:
	var fmt := RDTextureFormat.new()
	fmt.format       = RenderingDevice.DATA_FORMAT_R16G16B16A16_SFLOAT
	fmt.width        = grid_width
	fmt.height       = grid_height
	fmt.depth        = 1
	fmt.array_layers = 1
	fmt.mipmaps      = 1
	fmt.texture_type = RenderingDevice.TEXTURE_TYPE_2D
	fmt.usage_bits   = (
		RenderingDevice.TEXTURE_USAGE_SAMPLING_BIT   |
		RenderingDevice.TEXTURE_USAGE_STORAGE_BIT     |
		RenderingDevice.TEXTURE_USAGE_CAN_UPDATE_BIT  |
		RenderingDevice.TEXTURE_USAGE_CAN_COPY_FROM_BIT
	)

	tex_a = rd.texture_create(fmt, RDTextureView.new(), [])
	tex_b = rd.texture_create(fmt, RDTextureView.new(), [])
	tex_a_view = rd.texture_create_shared(RDTextureView.new(), tex_a)
	tex_b_view = rd.texture_create_shared(RDTextureView.new(), tex_b)


func _create_sampler() -> void:
	var smp := RDSamplerState.new()
	smp.min_filter = RenderingDevice.SAMPLER_FILTER_NEAREST
	smp.mag_filter = RenderingDevice.SAMPLER_FILTER_NEAREST
	smp.repeat_u   = RenderingDevice.SAMPLER_REPEAT_MODE_REPEAT
	smp.repeat_v   = RenderingDevice.SAMPLER_REPEAT_MODE_REPEAT
	sampler_rid = rd.sampler_create(smp)


func _compile_shaders() -> void:
	sim_shader   = _load_shader("res://ca_sim.glsl")
	brush_shader = _load_shader("res://ca_brush.glsl")
	sim_pipeline   = rd.compute_pipeline_create(sim_shader)
	brush_pipeline = rd.compute_pipeline_create(brush_shader)


func _create_uniform_sets() -> void:
	sim_set_ab = _make_sim_set(tex_a_view, tex_b)
	sim_set_ba = _make_sim_set(tex_b_view, tex_a)
	brush_set_a = _make_brush_set(tex_a)
	brush_set_b = _make_brush_set(tex_b)


func _make_sim_set(src_view: RID, dst_tex: RID) -> RID:
	var u0 := RDUniform.new()
	u0.uniform_type = RenderingDevice.UNIFORM_TYPE_SAMPLER_WITH_TEXTURE
	u0.binding = 0
	u0.add_id(sampler_rid)
	u0.add_id(src_view)

	var u1 := RDUniform.new()
	u1.uniform_type = RenderingDevice.UNIFORM_TYPE_IMAGE
	u1.binding = 1
	u1.add_id(dst_tex)

	return rd.uniform_set_create([u0, u1], sim_shader, 0)


func _make_brush_set(field_tex: RID) -> RID:
	var u0 := RDUniform.new()
	u0.uniform_type = RenderingDevice.UNIFORM_TYPE_IMAGE
	u0.binding = 0
	u0.add_id(field_tex)

	return rd.uniform_set_create([u0], brush_shader, 0)


func _process(_delta: float) -> void:
	_handle_brush_input()

	var compute_list := rd.compute_list_begin()

	if brush_active:
		_dispatch_brush(compute_list)

	for _i in range(steps_per_frame):
		_dispatch_simulation(compute_list)
		_swap_ping_pong()

	rd.compute_list_end()
	_update_output_texture()


func _dispatch_simulation(cl: int) -> void:
	var sim_set : RID = sim_set_ab if current_is_a else sim_set_ba

	rd.compute_list_bind_compute_pipeline(cl, sim_pipeline)
	rd.compute_list_bind_uniform_set(cl, sim_set, 0)

	# 24 floats = 96 bytes.  Order MUST match layout(push_constant) in ca_simulation.glsl.
	# Vulkan requires push constant size to be a multiple of 16 bytes (4 floats);

	var push := PackedFloat32Array([
		float(grid_width), float(grid_height), dt,
		mu_r, mu_g, mu_b,
		sigma_r, sigma_g, sigma_b,
		kernel_radius_r, kernel_radius_g, kernel_radius_b,
		coeff_rr, coeff_rg, coeff_rb,
		coeff_gr, coeff_gg, coeff_gb,
		coeff_br, coeff_bg, coeff_bb,
		0.0, 0.0, 0.0
	])

	rd.compute_list_set_push_constant(cl, push.to_byte_array(), push.size() * 4)

	var gx := ceili(float(grid_width) / 8.0)
	var gy := ceili(float(grid_height) / 8.0)
	rd.compute_list_dispatch(cl, gx, gy, 1)
	rd.compute_list_add_barrier(cl)


func _dispatch_brush(cl: int) -> void:
	var brush_set : RID = brush_set_a if current_is_a else brush_set_b

	rd.compute_list_bind_compute_pipeline(cl, brush_pipeline)
	rd.compute_list_bind_uniform_set(cl, brush_set, 0)

	var push := PackedFloat32Array([
		float(grid_width), float(grid_height),
		float(brush_pos.x), float(brush_pos.y),
		float(brush_radius),
		brush_color.r, brush_color.g, brush_color.b,
		1.0,
		0.0, 0.0, 0.0,
	])

	rd.compute_list_set_push_constant(cl, push.to_byte_array(), push.size() * 4)

	var gx := ceili(float(grid_width) / 8.0)
	var gy := ceili(float(grid_height) / 8.0)
	rd.compute_list_dispatch(cl, gx, gy, 1)
	rd.compute_list_add_barrier(cl)


func _swap_ping_pong() -> void:
	current_is_a = !current_is_a


func _update_output_texture() -> void:
	display_tex.texture_rd_rid = current_tex_rid()


func current_tex_rid() -> RID:
	return tex_a if current_is_a else tex_b


func _load_shader(path: String) -> RID:
	var file := load(path) as RDShaderFile
	if file == null:
		push_error("Failed to load RDShaderFile: %s" % path)
		return RID()

	var spirv := file.get_spirv()
	return rd.shader_create_from_spirv(spirv)


func _seed_random() -> void:
	var bytes := PackedByteArray()
	bytes.resize(grid_width * grid_height * 8)

	var ofs := 0
	for _i in range(grid_width * grid_height):
		bytes.encode_half(ofs + 0, randf() * seed_r)
		bytes.encode_half(ofs + 2, randf() * seed_g)
		bytes.encode_half(ofs + 4, randf() * seed_b)
		bytes.encode_half(ofs + 6, 1.0)
		ofs += 8

	rd.texture_update(tex_a, 0, bytes)


func _handle_brush_input() -> void:
	if Input.is_mouse_button_pressed(MOUSE_BUTTON_LEFT):
		brush_active = _set_brush_from_screen_pos(get_viewport().get_mouse_position())
	else:
		brush_active = false


func _unhandled_input(event: InputEvent) -> void:
	if event is InputEventScreenTouch or event is InputEventScreenDrag:
		var touch_pos : Vector2
		if event is InputEventScreenTouch:
			if not (event as InputEventScreenTouch).pressed:
				brush_active = false
				return
			touch_pos = (event as InputEventScreenTouch).position
		else:
			touch_pos = (event as InputEventScreenDrag).position
		brush_active = _set_brush_from_screen_pos(touch_pos)


func _set_brush_from_screen_pos(screen_pos: Vector2) -> bool:
	if output_rect == null:
		return false

	var canvas_xform := output_rect.get_global_transform_with_canvas()
	var inv := canvas_xform.affine_inverse()
	var local := inv * screen_pos
	var rect := output_rect.size

	if local.x < 0.0 or local.y < 0.0 or local.x > rect.x or local.y > rect.y:
		return false

	var uv := Vector2(local.x / rect.x, local.y / rect.y)
	brush_pos = Vector2i(
		clampi(int(uv.x * grid_width),  0, grid_width  - 1),
		clampi(int(uv.y * grid_height), 0, grid_height - 1)
	)
	return true


func _exit_tree() -> void:
	for rid : RID in [
		tex_a, tex_b, tex_a_view, tex_b_view,
		sampler_rid,
		sim_pipeline, brush_pipeline,
		sim_shader, brush_shader,
		sim_set_ab, sim_set_ba, brush_set_a, brush_set_b
	]:
		if rid.is_valid():
			rd.free_rid(rid)
