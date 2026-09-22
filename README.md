# Training and Scaling Language Models

Markdown source for the course website **Training and Scaling Language Models: From First
Principles to Efficient Serving**, designed by the [BRAIN team](https://www.imt-atlantique.fr/en/research-innovation/teams/brain) for [IMT Atlantique](https://www.imt-atlantique.fr/en).

The executable labs live in the public
[`brain-bzh/llm-course-companion`](https://github.com/brain-bzh/llm-course-companion)
repository. It is pinned here at `companion/` as a Git submodule for course
maintenance. Students should clone or fork the companion repository directly.

## Local development

```bash
git clone --recurse-submodules https://github.com/brain-bzh/llm-course.git
cd llm-course
uv sync
uv run mkdocs serve
```

Open <http://127.0.0.1:8000/llm-course/>.

For an existing course clone, initialize the companion with:

```bash
git submodule update --init
```

## Updating the companion

The submodule normally stays at the exact revision recorded by this repository.
To develop and publish a companion change, first switch the submodule from its
detached revision to `main`:

```bash
cd companion
git switch main
git pull --ff-only
# Edit, test, commit, and push the companion change.
cd ..
git add companion
# Commit the updated submodule pointer in this repository.
```

## Validation

```bash
uv run mkdocs build --strict
```

The generated static website is written to `site/`.

## Publishing

The workflow in `.github/workflows/pages.yml` builds the website from the
locked `uv` environment and publishes it through GitHub Pages on pushes to
`main`. Before the first deployment, configure **Settings → Pages → Source** as
**GitHub Actions** in the GitHub repository.
