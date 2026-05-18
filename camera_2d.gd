extends Camera2D

@export var speed := 300.0

func _process(delta):
	var v = Vector2(
		int(Input.is_key_pressed(KEY_D)) - int(Input.is_key_pressed(KEY_A)),
		int(Input.is_key_pressed(KEY_S)) - int(Input.is_key_pressed(KEY_W))
	).normalized()

	position += v * speed * delta

	if Input.is_key_pressed(KEY_Z): zoom *= 0.99
	if Input.is_key_pressed(KEY_X): zoom *= 1.01
