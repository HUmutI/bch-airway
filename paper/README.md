# paper/

LaTeX source and bibliography of the method paper, *Airway Segmentation and
Anatomical Branch Assignment in Pediatric Chest CT*. Figures are referenced
as `figures/*.pdf` and are the renders in `../evaluation/FIGURES/`:

```bash
cd paper && ln -s ../evaluation/FIGURES figures && pdflatex airway_method_paper && bibtex airway_method_paper && pdflatex airway_method_paper && pdflatex airway_method_paper
```

The numbers in the tables come from `../evaluation/` (adult and pediatric
benchmarks, rule sweep); `evaluation/README.md` says which script prints
which table. Working PDFs, earlier drafts and the supplementary bundles stay
on izmir in `paper/`.
