# Makefile for Training Language Models Lectures
# Usage:
#   make all            - Compile all 13 lectures (twice for TOC references)
#   make session-01     - Compile only session 01
#   make clean          - Remove LaTeX auxiliary files
#   make distclean      - Remove LaTeX auxiliary files and compiled PDFs

SESSIONS = 01-transformer \
           02-training-loop \
           03-data-pipeline \
           04-small-gpt \
           05-data-selection \
           06-single-gpu \
           07-ddp \
           08-fsdp \
           09-tensor-parallelism \
           10-context-pipeline \
           11-kv-cache \
           12-serving \
           13-beyond-transformers

PDFS = $(foreach s,$(SESSIONS),$(s)/$(s).pdf)

.PHONY: all clean distclean $(SESSIONS)

all: $(PDFS)

define SESSION_RULE
$(1)/$(1).pdf: $(1)/$(1).tex common/preamble.tex common/style.sty common/colors.tex
	@echo "==> Compiling $(1)..."
	@cd $(1) && pdflatex -interaction=nonstopmode $(1).tex > /dev/null
	@cd $(1) && pdflatex -interaction=nonstopmode $(1).tex > /dev/null
	@echo "✓ Generated $(1)/$(1).pdf"

$(1): $(1)/$(1).pdf
endef

$(foreach s,$(SESSIONS),$(eval $(call SESSION_RULE,$(s))))

clean:
	@echo "Cleaning auxiliary LaTeX files..."
	@find . -type f \( -name "*.aux" -o -name "*.log" -o -name "*.nav" -o -name "*.out" -o -name "*.snm" -o -name "*.toc" -o -name "*.vrb" \) -delete
	@echo "Done."

distclean: clean
	@echo "Cleaning generated PDFs..."
	@find . -type f -name "*.pdf" ! -path "*/common/*" -delete
	@echo "Done."
