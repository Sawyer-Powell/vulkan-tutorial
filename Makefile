COMP_SHADER = src/shaders/shader.comp
VERT_SHADER = src/shaders/shader.vert
FRAG_SHADER = src/shaders/shader.frag

shaders: $(COMP_SHADER) $(VERT_SHADER) $(FRAG_SHADER)
	glslc $(COMP_SHADER) -o comp.spv
	glslc $(VERT_SHADER) -o vert.spv
	glslc $(FRAG_SHADER) -o frag.spv

prod: Cargo.toml shaders
	cargo build --release

run: shaders
	cargo run

test: shaders
	cargo test -- --nocapture

.PHONY: shaders prod run
