# Glossary

**Mode**: what a generation does with its inputs: **txt2img** (prompt → new image), **img2img** (an image + prompt → a changed image), **inpaint** (an image + mask + prompt → the masked area repainted).
_Avoid_: pipeline (in diffusers, a pipeline is the loaded model object; older code calls a mode `pipeline_name`)

**Model**: a named image generation model such as Z-Image Turbo. Identified by a **slug** like `z-image-turbo` in the MCP server.
_Avoid_: version (older code calls a model `version`)

**Mask**: the image that selects the area to change in inpaint. White = repaint, black = keep.

**Strength**: how much img2img and inpaint change the input image. Ignored by FLUX.2 [klein] in img2img.

**Seed**: a number that makes a generation reproducible. Random when unset.

**Settings file**: the .txt saved next to each output image, recording the settings used (including the seed).

**CPU offload**: moving parts of a model to system RAM to fit GPU memory. Automatic below 24GB for Z-Image and FLUX.2.
