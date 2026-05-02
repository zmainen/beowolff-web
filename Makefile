HAAK := $(HOME)/Projects/haak

# SSOT sources in HAAK → derived files in this repo
DERIVED := research/articles/00-art-genome-taxonomy.md

.PHONY: build deploy clean

build: $(DERIVED)

# Art Genome taxonomy: strip YAML frontmatter, mark as generated
research/articles/00-art-genome-taxonomy.md: $(HAAK)/home/commons/datasets/art-genome/taxonomy.md
	@echo "  SYNC  $< → $@"
	@printf '<!-- GENERATED — do not edit. SSOT: %s -->\n\n' '$<' > $@
	@sed '1{/^---$$/!q;};1,/^---$$/d' $< | sed '1{/^$$/d;}' >> $@

deploy: build
	git add -A
	git status --short
	@echo "---"
	@echo "Review above, then: git commit && git push"

clean:
	rm -f $(DERIVED)
