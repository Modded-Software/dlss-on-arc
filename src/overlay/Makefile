ROOT ?= $(abspath ../..)
OUT ?= $(ROOT)/work
BIN := $(OUT)/gpu_overlay
PREFIX ?= /usr/local
BINDIR ?= $(PREFIX)/bin
DESTDIR ?=
RUN_ARGS ?= --draggable

VK_INC ?= $(ROOT)/work/vulkan-headers/include
X11_INC ?= $(ROOT)/work/deps/x11/usr/include
GLSLANG ?= $(firstword $(wildcard $(ROOT)/work/deps/glslang/bin/glslangValidator) $(shell command -v glslangValidator) glslangValidator)
X11_SO ?= $(firstword $(wildcard /usr/lib/x86_64-linux-gnu/libX11.so.6) $(wildcard /usr/lib/aarch64-linux-gnu/libX11.so.6))
VULKAN_SO ?= $(firstword $(wildcard /usr/lib/x86_64-linux-gnu/libvulkan.so.1) $(wildcard /usr/lib/aarch64-linux-gnu/libvulkan.so.1))
X11_LIB ?= $(if $(X11_SO),$(X11_SO),-lX11)
VULKAN_LIB ?= $(if $(VULKAN_SO),$(VULKAN_SO),-lvulkan)

ifeq ($(origin CC),default)
CC := $(firstword $(wildcard /usr/bin/gcc) $(shell command -v gcc) cc)
endif
CFLAGS ?= -O2 -g -Wall -Wextra -Wno-unused-parameter -std=c11
CPPFLAGS += -I$(VK_INC) -I$(X11_INC) -I$(OUT)
LDFLAGS += -B/usr/bin
LDLIBS += $(VULKAN_LIB) $(X11_LIB) -ldl -lm -lpthread

all: build

build: $(BIN)

$(OUT):
	mkdir -p $(OUT)

$(OUT)/overlay_vert.h: overlay.vert | $(OUT)
	$(GLSLANG) -V --target-env vulkan1.1 --variable-name overlay_vert_spv -o $@ $<

$(OUT)/overlay_frag.h: overlay.frag | $(OUT)
	$(GLSLANG) -V --target-env vulkan1.1 --variable-name overlay_frag_spv -o $@ $<

$(BIN): gpu_overlay.c $(OUT)/overlay_vert.h $(OUT)/overlay_frag.h Makefile | $(OUT)
	$(CC) $(CFLAGS) $(CPPFLAGS) $(LDFLAGS) -o $@ gpu_overlay.c $(LDLIBS)

install: $(BIN)
	install -d $(DESTDIR)$(BINDIR)
	install -m755 $(BIN) $(DESTDIR)$(BINDIR)/gpu_overlay

uninstall:
	rm -f $(DESTDIR)$(BINDIR)/gpu_overlay

deploy: $(BIN)
	install -d $(HOME)/.local/bin
	install -m755 $(BIN) $(HOME)/.local/bin/gpu_overlay

run: $(BIN)
	$(BIN) $(RUN_ARGS)

clean:
	rm -f $(BIN) $(OUT)/overlay_vert.h $(OUT)/overlay_frag.h

.PHONY: all build install uninstall deploy run clean
