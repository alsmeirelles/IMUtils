# Makefile for IMUtils
#
# Common usage:
#   make sync
#   make test
#   make build
#   make clean-build

UV ?= uv
PYTEST_ARGS ?=

DIST_DIR := dists

.PHONY: sync test build clean-build

sync:
	$(UV) sync --all-extras --dev

test:
	$(UV) run pytest $(PYTEST_ARGS)

build: clean-build
	mkdir -p $(DIST_DIR)
	$(UV) build --out-dir $(DIST_DIR)

clean-build:
	rm -rf $(DIST_DIR)
	rm -rf build
	rm -rf *.egg-info
	rm -rf src/*.egg-info