#!/bin/sh
# Derives research/articles/00-art-genome-taxonomy.md from the HAAK SSOT.
# SSOT: ~/Projects/haak/home/commons/datasets/art-genome/taxonomy.md
# Run before committing web changes that touch the taxonomy.
set -e
SSOT="$HOME/Projects/haak/home/commons/datasets/art-genome/taxonomy.md"
OUT="$(dirname "$0")/research/articles/00-art-genome-taxonomy.md"
[ -f "$SSOT" ] || { echo "SSOT not found: $SSOT" >&2; exit 1; }
# Strip YAML frontmatter, prepend generated notice
sed '1{/^---$/!q;};1,/^---$/d' "$SSOT" | sed '1{/^$/d;}' > "$OUT"
sed -i '' '1i\
<!-- GENERATED — do not edit. SSOT: haak/home/commons/datasets/art-genome/taxonomy.md -->\
' "$OUT"
echo "Synced: $SSOT → $OUT ($(wc -l < "$OUT" | tr -d ' ') lines)"
