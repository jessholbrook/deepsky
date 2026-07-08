---
title: deepsky
emoji: 🌌
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: 4.44.1
app_file: app.py
pinned: false
license: cc-by-4.0
models:
  - jessholbrook/deepsky-128px
---

# deepsky — live demo

A from-scratch denoising diffusion model that generates deep-sky astronomical
images (nebulae, galaxies, star clusters), trained on public ESA/Hubble,
ESA/Webb, ESO, and NASA imagery.

- **Model:** [jessholbrook/deepsky-128px](https://huggingface.co/jessholbrook/deepsky-128px)
- **Code + write-up:** [github.com/jessholbrook/deepsky](https://github.com/jessholbrook/deepsky)

Press **Generate** to sample a fresh 128px image. On the free CPU tier each image
takes ~30–90s (DDIM sampling); lower the step count for faster (softer) results,
or upgrade the Space hardware to a GPU for near-instant generation.

These files are mirrored from the [`space/`](https://github.com/jessholbrook/deepsky/tree/main/space)
directory of the main repo.
