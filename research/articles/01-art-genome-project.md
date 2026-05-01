# The Art Genome Project

Artsy is the largest online art marketplace, and one of the few platforms in the art world to open-source significant portions of its technology stack. In April 2026, Artsy merged with Artnet under Beowolff Capital, creating the dominant player in online art commerce. Their most distinctive intellectual contribution is not their marketplace software but their classification system: the Art Genome Project, a controlled vocabulary of 1,031 named categories for describing art.

The genome is Artsy's answer to the question every art platform must eventually face: how do you organize a collection so that someone looking at a Rothko can find their way to a Newman, or a Richter, or a contemporary Chinese painter working in a similar register? Library classification systems (LC, Dewey) organize by subject matter. Museum taxonomies organize by period, region, and medium. The Art Genome Project attempts something more ambitious — a multi-dimensional classification that captures visual qualities, historical context, technique, materials, and subject matter within a single framework.

## Structure

The genome consists of 1,031 genes organized into 17 families. Each gene is a named category — "Impasto," "Geometric," "Bright and Vivid Colors," "Abstract Expressionism" — that can be applied to an artwork with a relevance score from 1 to 100. The gene "Chiaroscuro" might be applied to a Caravaggio at 98 and to a de La Tour at 85 and not applied at all to a Mondrian. The taxonomy and its descriptions are published as a CSV file under CC-BY-4.0; the scored applications to individual artworks are not public.

Each gene entry has six fields:

| Field | Example |
|:------|:--------|
| ID | `4d90d18fdcdd5f44a500002d` |
| Slug | `art-informel` |
| Name | Art Informel |
| Family | Styles and Movements |
| Description | Markdown text (~100–700 words) |
| Automated | TRUE or FALSE |

The `automated` flag is the most revealing field. It indicates whether a gene is applied algorithmically or by a human "genomer" — Artsy's term for the specialists who evaluate artworks against the full vocabulary. Of the 1,031 genes, 748 are manually applied. The 283 automated genes are overwhelmingly geographic regions, time periods, and cultural styles — categories where algorithmic assignment from metadata is reliable. The perceptually interesting categories are almost entirely human work.

## The 17 families

| Family | Genes | Manual | Automated |
|:-------|------:|-------:|----------:|
| Styles and Movements | 251 | 190 | 61 |
| Subject Matter | 213 | 201 | 12 |
| Medium and Techniques | 175 | 162 | 13 |
| Visual Qualities | 74 | 71 | 3 |
| Cultural and Religious Styles | 69 | 5 | 64 |
| Geographic Regions | 67 | 2 | 65 |
| Materials | 39 | 30 | 9 |
| Furniture and Lighting | 27 | 18 | 9 |
| Design Movements | 26 | 10 | 16 |
| Design Concepts and Techniques | 22 | 17 | 5 |
| Time Periods | 20 | 5 | 15 |
| Tableware, Vessels, and Objects | 17 | 13 | 4 |
| Antiquities, Artifacts, Religious Objects | 7 | 7 | 0 |
| Artistic Disciplines | 7 | 6 | 1 |
| Jewelry and Fashion Object Types | 7 | 7 | 0 |
| Textiles | 6 | 3 | 3 |

The distribution reveals a design philosophy. "Styles and Movements" (251 genes) is by far the largest family — nearly a quarter of the entire taxonomy. It covers everything from "Abstract Expressionism" to "Young British Artists," from "Bauhaus" to "Vaporwave." This is the vocabulary of art history as practiced in galleries and auction houses: movements, schools, -isms. It is how the art market talks about art.

"Subject Matter" (213 genes) is the second largest. It includes both concrete subjects ("Dogs," "Horses," "Flowers") and conceptual themes ("Memory," "Identity and the Body," "Globalization"). "Medium and Techniques" (175 genes) covers both materials ("Oil Paint," "Bronze") and processes ("Impasto," "Screenprint").

The family that matters most for perceptual similarity — "Visual Qualities" — has only 74 genes. The genome was designed for a marketplace, and marketplaces organize by what buyers search for: movement, period, medium, subject. Visual appearance is a secondary axis.

## The genomer process

Artsy describes the Art Genome Project as a "dynamic controlled vocabulary" with active management. Genes are "re-defined, added, renamed, merged and eliminated often." The genomers — Artsy's classification specialists — evaluate artworks individually, applying relevant genes with 1-to-100 relevance scores. This is labor-intensive, expert-dependent work. Each artwork in Artsy's collection is positioned in a 1,031-dimensional space defined by these scores.

The 1-to-100 scale introduces a crucial distinction between the genome and simpler tagging systems. A binary tag says "this work is Impressionist" or "this work is not Impressionist." A scored gene says "this work is Impressionist to degree 73." Monet's *Impression, Sunrise* might score 95 on Impressionism. A late Renoir might score 60 — Impressionist in technique but moving toward something else. A Whistler might score 30 — influenced by Impressionism but belonging to a different tradition. The scores create a continuous landscape rather than a discrete partition.

## What is public, what is not

The open-source repository contains the vocabulary: 1,031 gene definitions with names, families, descriptions, and automation flags. It does not contain the scored applications — which artworks carry which genes at which scores. That data lives in Artsy's internal database.

The Artsy public API (v2) can return which genes are associated with a given artwork, but it suppresses the numeric scores. You can learn that a Rothko is tagged with "Color Field Painting" and "Abstract Expressionism" and "Bright and Vivid Colors," but you cannot learn the 1-to-100 scores that quantify the strength of each association.

The open-source gene data was last updated in October 2018. The internal genome has almost certainly continued evolving — genes redefined, merged, added, eliminated — but those changes have not been published. The public export is a snapshot of a living system, frozen seven years ago.

This asymmetry — public vocabulary, private scores — is characteristic of Artsy's open-source strategy generally. They publish the structure, not the substance. The schema is open; the data is not. The vocabulary is open; the judgments are not. This pattern recurs across their entire technology stack, as we will see in later chapters.

---

*Source: [github.com/artsy/the-art-genome-project](https://github.com/artsy/the-art-genome-project) (CC-BY-4.0). Gene counts and automation flags from `genes.csv`, surveyed May 2026.*
