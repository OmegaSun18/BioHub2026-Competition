DATA ?= /data

install:           ; pip install -e ".[dev]"
test:              ; pytest -q
# train + evaluate one variant:   make run V=raw   (or temporal_stack, temporal_diff)
run:               ; biohub train --config configs/$(V).yaml --set data.root=$(DATA)
