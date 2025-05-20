all: lib

lib: check_submodule config
	echo afxdp/lib; cd afxdp; $(MAKE) -C lib
	echo afxdp/lib/xdp-tools; cd afxdp; $(MAKE) -C lib/xdp-tools

check_submodule:
	git submodule update --init

config: afxdp/configure
	cd afxdp; ./configure
	cd afxdp/lib/xdp-tools; ./configure

clean: $(LESSONS_CLEAN)
	echo afxdp/lib/xdp-tools; cd afxdp; $(MAKE) -C lib/xdp-tools clean
	echo afxdp/lib; cd afxdp; $(MAKE) -C lib clean
	echo afxdp/common; cd afxdp; $(MAKE) -C common clean
