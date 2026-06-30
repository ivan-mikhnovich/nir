.PHONY: all clean view

TEX := slides.tex
PDF := slides.pdf
LATEX := lualatex
LATEX_FLAGS := -interaction=nonstopmode -halt-on-error

IMAGES_SCRIPT := generate_images.py

all: $(PDF)

# Генерируем изображения перед компиляцией (скрипт сам проверяет кеш).
$(PDF): $(TEX) $(IMAGES_SCRIPT)
	uv run $(IMAGES_SCRIPT)
	$(LATEX) $(LATEX_FLAGS) $(TEX)
	$(LATEX) $(LATEX_FLAGS) $(TEX)

view: $(PDF)
	zathura $(PDF) 2>/dev/null || xdg-open $(PDF) 2>/dev/null || open $(PDF) 2>/dev/null || true

clean:
	rm -f *.aux *.log *.nav *.out *.snm *.toc *.vrb
	rm -f $(PDF)
