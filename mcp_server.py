"""A streamable HTTP MCP server for generating and editing images with the models in sd2/generate.py.

Run with `uv run python mcp_server.py [--port 8765] [--preload <slug>]`.
It only listens on 127.0.0.1 and has no authentication.
"""

import argparse
import asyncio
import io
import json
import os

import anyio
import anyio.from_thread
import anyio.to_thread
import torch
from mcp.server.mcpserver import Context, Image as McpImage, MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from PIL import Image, UnidentifiedImageError

from sd2 import generate as gen

HOST = "127.0.0.1"
DEFAULT_PORT = 8765
THUMBNAIL_SIZE = 256
MIN_SIZE, MAX_SIZE = 64, 2048
# Next to this file, like the app's outputs/ when run from the repo root.
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")

# Mode -> the tool that runs it.
TOOL_FOR_MODE = {"txt2img": "generate_image", "img2img": "edit_image", "inpaint": "inpaint"}

mcp = MCPServer(
    "diffusers",
    instructions=(
        "Generate and edit images with local diffusion Models. Call list_models to see the Models, "
        "their defaults and which tools each supports. Every result is saved as a PNG (with a .txt of "
        "the settings) and returns its path plus a small thumbnail. Calls are queued: one image is "
        "generated at a time, and the first call with a Model loads it, which can take a while."
    ),
)

# One generation at a time: the GPU only fits one pipeline.
_lock = asyncio.Lock()


def _models_supporting(mode):
    return [slug for slug, version in gen.MODEL_SLUGS.items() if (mode, version) in gen.PIPELINES]


def _resolve_model(model, mode):
    version = gen.MODEL_SLUGS.get(model)
    if version is None:
        raise ToolError(f"Unknown model '{model}'. Available models: {', '.join(gen.MODEL_SLUGS)}.")
    if (mode, version) not in gen.PIPELINES:
        raise ToolError(
            f"Model '{model}' does not support {TOOL_FOR_MODE[mode]} ({mode}). "
            f"Models that do: {', '.join(_models_supporting(mode))}."
        )
    return version


def _check_size(width, height):
    for name, value in (("width", width), ("height", height)):
        if not MIN_SIZE <= value <= MAX_SIZE or value % 16:
            raise ToolError(
                f"Invalid {name} {value}: it must be a multiple of 16 between {MIN_SIZE} and {MAX_SIZE}."
            )


def _check_common(steps, guidance_scale, strength=None, seed=None):
    if steps is not None and steps < 1:
        raise ToolError(f"Invalid steps {steps}: it must be at least 1.")
    if guidance_scale is not None and guidance_scale < 0:
        raise ToolError(f"Invalid guidance_scale {guidance_scale}: it must be 0 or more.")
    if strength is not None and not 0 <= strength <= 1:
        raise ToolError(f"Invalid strength {strength}: it must be between 0 and 1.")
    if seed is not None and not 0 <= seed < 2**64:
        raise ToolError(f"Invalid seed {seed}: it must be a whole number from 0 to 2**64-1.")


def _load_image(path, what, mode="RGB"):
    full_path = os.path.abspath(os.path.expanduser(path))
    try:
        with Image.open(full_path) as img:
            return img.convert(mode)
    except FileNotFoundError:
        raise ToolError(f"The {what} file does not exist: {full_path}")
    except (UnidentifiedImageError, OSError) as e:
        raise ToolError(f"Could not read the {what} file {full_path} as an image: {e}")


def _thumbnail(image):
    thumb = image.convert("RGB")
    thumb.thumbnail((THUMBNAIL_SIZE, THUMBNAIL_SIZE))
    buffer = io.BytesIO()
    thumb.save(buffer, format="PNG")
    return McpImage(data=buffer.getvalue(), format="png")


async def _run(ctx, mode, model, prompt, *, steps, guidance_scale, negative_prompt, seed, **kwargs):
    """Runs one generation in a worker thread, queued behind the lock."""
    version = _resolve_model(model, mode)
    default_steps, default_guidance = gen.MODEL_DEFAULTS.get(version, gen.DEFAULT_STEPS_AND_GUIDANCE)
    steps = default_steps if steps is None else steps
    guidance_scale = default_guidance if guidance_scale is None else guidance_scale

    def on_progress(step, total):
        # Called from the worker thread, after each denoising step.
        try:
            anyio.from_thread.run(ctx.report_progress, step + 1, total, f"Step {step + 1}/{total}")
        except Exception:
            pass  # Progress is best effort.

    async with _lock:
        try:
            result = await anyio.to_thread.run_sync(
                lambda: gen.generate(
                    prompt,
                    mode,
                    negative_prompt=negative_prompt,
                    steps=steps,
                    guidance_scale=guidance_scale,
                    version=version,
                    seed=seed,
                    on_progress=on_progress,
                    output_dir=OUTPUT_DIR,
                    **kwargs,
                )
            )
        except torch.cuda.OutOfMemoryError:
            gen.unload_pipeline()
            raise ToolError(
                "The GPU ran out of memory. The model was unloaded; try again, or use a smaller "
                "width/height, or a smaller model (see list_models)."
            )

    info = {
        "path": result.path,
        "model": model,
        "mode": mode,
        "prompt": prompt,
        "negative_prompt": negative_prompt or None,
        "seed": result.seed,
        "width": result.size[0],
        "height": result.size[1],
        "steps": steps,
        "guidance_scale": guidance_scale,
        "cpu_offload": result.enable_cpu_offload,
        # Settings the Model doesn't use, e.g. negative_prompt for FLUX.2.
        "ignored": result.ignored,
        "settings_path": os.path.splitext(result.path)[0] + ".txt",
    }
    if mode != "txt2img":
        info["strength"] = kwargs["strength"]
    return [json.dumps(info, indent=2), _thumbnail(result.image)]


@mcp.tool()
def list_models() -> list[dict]:
    """List the Models: slug (use it as the `model` argument), display name, Hugging Face repo, the tools each supports
    (generate_image, edit_image, inpaint), default steps/guidance_scale/size and a usage hint."""
    models = []
    for slug, version in gen.MODEL_SLUGS.items():
        steps, guidance = gen.MODEL_DEFAULTS.get(version, gen.DEFAULT_STEPS_AND_GUIDANCE)
        width, height = gen.MODEL_SIZES.get(version, gen.DEFAULT_SIZE)
        modes = {mode: gen.PIPELINES[(mode, version)][1] for mode in TOOL_FOR_MODE if (mode, version) in gen.PIPELINES}
        models.append(
            {
                "slug": slug,
                "name": version,
                "repo": next(iter(modes.values())),
                "tools": [TOOL_FOR_MODE[mode] for mode in modes],
                "default_steps": steps,
                "default_guidance_scale": guidance,
                "default_width": width,
                "default_height": height,
                "hint": gen.MODEL_HINTS.get(version, ""),
            }
        )
    return models


@mcp.tool()
async def generate_image(
    prompt: str,
    ctx: Context,
    model: str = "z-image-turbo",
    width: int | None = None,
    height: int | None = None,
    steps: int | None = None,
    guidance_scale: float | None = None,
    negative_prompt: str | None = None,
    seed: int | None = None,
) -> list:
    """Generate an image from a text prompt (txt2img). Supported by z-image-turbo, flux2-klein-4b, sdxl and sdxl-turbo.

    The image is saved as a PNG; the result has its absolute `path`, the settings used (including the seed) and a
    small thumbnail. Calls are queued, and the first use of a Model loads it (slow).

    Args:
        prompt: What to draw.
        model: Model slug, see list_models. Default z-image-turbo.
        width: Width in pixels, a multiple of 16 (64-2048). Default: the Model's native width (1024, sdxl-turbo 512).
        height: Height in pixels, same rules as width.
        steps: Number of inference steps. Default: the Model's own (e.g. 9 for z-image-turbo, 4 for flux2-klein-4b).
        guidance_scale: How closely to follow the prompt. Default: the Model's own (0 for the turbo Models).
        negative_prompt: What to avoid. Ignored by Models that don't support it (flux2-klein-4b, guidance 0 Models).
        seed: Whole number for reproducible results. Default: random; the seed used is in the result.
    """
    version = _resolve_model(model, "txt2img")
    default_width, default_height = gen.MODEL_SIZES.get(version, gen.DEFAULT_SIZE)
    width = default_width if width is None else width
    height = default_height if height is None else height
    _check_size(width, height)
    _check_common(steps, guidance_scale, seed=seed)
    return await _run(
        ctx, "txt2img", model, prompt,
        steps=steps, guidance_scale=guidance_scale, negative_prompt=negative_prompt, seed=seed,
        width=width, height=height,
    )  # fmt: skip


@mcp.tool()
async def edit_image(
    image_path: str,
    prompt: str,
    ctx: Context,
    model: str = "z-image-turbo",
    strength: float = 0.3,
    steps: int | None = None,
    guidance_scale: float | None = None,
    negative_prompt: str | None = None,
    seed: int | None = None,
) -> list:
    """Edit or restyle an existing image from a prompt (img2img). Supported by z-image-turbo, flux2-klein-4b and
    sdxl-refiner (the refiner only polishes details).

    The output keeps the input's aspect ratio (z-image-turbo and flux2-klein-4b round the size down to a multiple of
    16). The result has the saved PNG's absolute `path`, the settings used (including the seed) and a small thumbnail.

    Args:
        image_path: Local path of the input image (png or jpg).
        prompt: For flux2-klein-4b, describe the change ("replace the dog with a cat"); otherwise describe the result.
        model: Model slug, see list_models. Default z-image-turbo.
        strength: 0-1, how much to change the image: low stays close to it, 1 ignores it. Default 0.3.
            Ignored by flux2-klein-4b.
        steps: Number of inference steps. Default: the Model's own. With a strength below 1 only part of them run.
        guidance_scale: How closely to follow the prompt. Default: the Model's own.
        negative_prompt: What to avoid. Ignored by Models that don't support it.
        seed: Whole number for reproducible results. Default: random; the seed used is in the result.
    """
    _resolve_model(model, "img2img")
    _check_common(steps, guidance_scale, strength, seed)
    image = _load_image(image_path, "image")
    return await _run(
        ctx, "img2img", model, prompt,
        steps=steps, guidance_scale=guidance_scale, negative_prompt=negative_prompt, seed=seed,
        image_input=image, strength=strength,
    )  # fmt: skip


@mcp.tool()
async def inpaint(
    image_path: str,
    mask_path: str,
    prompt: str,
    ctx: Context,
    model: str = "z-image-turbo",
    strength: float = 1.0,
    steps: int | None = None,
    guidance_scale: float | None = None,
    negative_prompt: str | None = None,
    seed: int | None = None,
) -> list:
    """Repaint part of an image from a prompt (inpainting). Supported by z-image-turbo, flux2-klein-4b and sdxl.

    The mask marks what to repaint: WHITE pixels are repainted, black pixels are kept. It is converted to grayscale
    and, if its size differs from the image's, resized to it. The result has the saved PNG's absolute `path`, the
    settings used (including the seed) and a small thumbnail.

    Args:
        image_path: Local path of the input image (png or jpg).
        mask_path: Local path of the mask image (white = repaint).
        prompt: What to paint in the masked area. For flux2-klein-4b, describe the change.
        model: Model slug, see list_models. Default z-image-turbo.
        strength: 0-1, how much the masked area changes: 1 (default) essentially ignores the original there.
        steps: Number of inference steps. Default: the Model's own.
        guidance_scale: How closely to follow the prompt. Default: the Model's own.
        negative_prompt: What to avoid. Ignored by Models that don't support it.
        seed: Whole number for reproducible results. Default: random; the seed used is in the result.
    """
    _resolve_model(model, "inpaint")
    _check_common(steps, guidance_scale, strength, seed)
    image = _load_image(image_path, "image")
    mask = _load_image(mask_path, "mask", mode="L")
    if mask.size != image.size:
        mask = mask.resize(image.size, Image.NEAREST)
    mask = mask.point(lambda v: 255 if v > 127 else 0)
    return await _run(
        ctx, "inpaint", model, prompt,
        steps=steps, guidance_scale=guidance_scale, negative_prompt=negative_prompt, seed=seed,
        image_input=image, mask_input=mask, strength=strength,
    )  # fmt: skip


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--preload", choices=list(gen.MODEL_SLUGS), help="load this Model's pipeline at startup")
    args = parser.parse_args()

    if args.preload:
        version = gen.MODEL_SLUGS[args.preload]
        mode = next(m for m in TOOL_FOR_MODE if (m, version) in gen.PIPELINES)  # txt2img if supported
        print(f"Preloading {args.preload} ({mode})")
        gen.get_pipeline(mode, version=version, enable_cpu_offload=gen.needs_cpu_offload(version))

    mcp.run("streamable-http", host=HOST, port=args.port)


if __name__ == "__main__":
    main()
