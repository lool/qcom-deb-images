# DEBOS_OPTS can be overridden with:
#     make DEBOS_OPTS=... all

# To build large images, the debos resource defaults are not sufficient. These
# provide defaults that work for us as universally as we can manage.
FAKEMACHINE_BACKEND = $(shell [ -c /dev/kvm ] && echo kvm || echo qemu)
DEBOS_OPTS := --fakemachine-backend $(FAKEMACHINE_BACKEND) --memory 1GiB --scratchsize 4GiB
DEBOS := debos $(DEBOS_OPTS)

# Use http_proxy from the environment, or apt's http_proxy if set, to speed up
# builds.
http_proxy ?= $(shell apt-config dump --format '%v%n' Acquire::http::Proxy)
export http_proxy

all: disk-ufs.img.gz disk-sdcard.img.gz

rootfs.tar.gz: debos-recipes/qualcomm-linux-debian-rootfs.yaml
	$(DEBOS) $<

disk-ufs.img disk-ufs.img.gz: debos-recipes/qualcomm-linux-debian-image.yaml rootfs.tar.gz
	$(DEBOS) $<

disk-sdcard.img.gz: debos-recipes/qualcomm-linux-debian-image.yaml rootfs.tar.gz
	$(DEBOS) -t imagetype:sdcard $<

test: disk-ufs.img
	# rootfs/ is a build artifact, so should not be scanned for tests
	py.test-3 --ignore=rootfs

.PHONY: all test
