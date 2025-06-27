# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD  ?= sphinx-build
SOURCEDIR    = source
BUILDDIR     = build

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

# Custom targets for GLI documentation
install-deps:
	pip install sphinx sphinx-rtd-theme myst-parser

clean-all:
	rm -rf $(BUILDDIR)/*
	rm -rf source/api/generated/

dev-build:
	@$(SPHINXBUILD) -b html "$(SOURCEDIR)" "$(BUILDDIR)/html" $(SPHINXOPTS) $(O) -W

watch:
	@echo "Watching for changes... Press Ctrl+C to stop."
	@while true; do \
		$(SPHINXBUILD) -b html "$(SOURCEDIR)" "$(BUILDDIR)/html" $(SPHINXOPTS) $(O) -q; \
		echo "Documentation built at $$(date)"; \
		sleep 5; \
	done

serve:
	@echo "Starting local server at http://localhost:8000"
	@cd $(BUILDDIR)/html && python -m http.server 8000

linkcheck-fast:
	@$(SPHINXBUILD) -b linkcheck "$(SOURCEDIR)" "$(BUILDDIR)/linkcheck" $(SPHINXOPTS) $(O) -q

# Build documentation with different themes for testing
build-themes:
	@$(SPHINXBUILD) -b html "$(SOURCEDIR)" "$(BUILDDIR)/html-rtd" $(SPHINXOPTS) $(O) -D html_theme=sphinx_rtd_theme
	@$(SPHINXBUILD) -b html "$(SOURCEDIR)" "$(BUILDDIR)/html-alabaster" $(SPHINXOPTS) $(O) -D html_theme=alabaster
	@echo "Built documentation with multiple themes in $(BUILDDIR)/"
