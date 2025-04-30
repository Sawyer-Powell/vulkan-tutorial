VERT_SHADER = src/shaders/shader.vert
FRAG_SHADER = src/shaders/shader.frag

shaders: $(VERT_SHADER) $(FRAG_SHADER)
	glslc $(VERT_SHADER) -o vert.spv
	glslc $(FRAG_SHADER) -o frag.spv

prod: Cargo.toml *.rs shaders
	cargo build --production

run: shaders
	cargo run

test: shaders
	cargo test -- --nocapture

.PHONY: shaders prod run
