# Repository guidance

## Course and companion repositories

- This repository contains the course website, lectures, lab guides, and
  assessment material.
- Executable course code is maintained in the public
  [`brain-bzh/llm-course-companion`](https://github.com/brain-bzh/llm-course-companion)
  repository, included here at `companion/` as a Git submodule.
- Make companion-code changes inside the submodule. Commit and push them to the
  companion repository first, then commit the updated `companion` pointer in
  this repository. Do not duplicate companion source files in the course
  repository.
- Students clone or fork the companion repository directly; the submodule is
  primarily for course maintainers.

## SVG figures

- Keep SVGs visually restrained and diagram-first. Use short labels rather than
  paragraphs or dense explanatory text.
- Put detailed explanations in the surrounding Markdown instead of inside the
  figure.
- Render and inspect every changed SVG. Check that labels remain legible and
  that no text overlaps or is clipped at the intended display size.
