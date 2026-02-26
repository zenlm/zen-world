# Zen World

Immersive 3D world generation model from [Zen LM](https://zenlm.org).

Zen World generates explorable, interactive 3D environments from text and image inputs. It combines panoramic world representation with semantic-aware scene decomposition to produce coherent, navigable 3D worlds.

## Capabilities

- **Text-to-3D World**: Generate full 360° environments from text descriptions
- **Image-to-3D World**: Lift a single image into an explorable 3D scene
- **Mesh Export**: Export scenes as standard mesh formats for use in game engines and graphics pipelines
- **Object Decomposition**: Disentangled foreground/background representations for interactive scenes
- **Virtual Reality**: VR-ready immersive outputs
- **Game Development**: Mesh + semantic structure compatible with Unity, Unreal, and Godot

## Model Variants

| Variant | Size | Use Case |
|---------|------|----------|
| **zen-world** | Full precision | Production rendering |
| **zen-world-lite** | Quantized | Consumer GPU (16GB+ VRAM) |

## Quick Start

### Requirements

```bash
pip install zen-world torch torchvision diffusers
```

Hardware requirements:
- **zen-world**: 2x A100 80GB recommended
- **zen-world-lite**: Single GPU with 16GB+ VRAM

### Text to 3D World

```python
from zen_world import ZenWorldPipeline

pipeline = ZenWorldPipeline.from_pretrained("zenlm/zen-world")
pipeline = pipeline.to("cuda")

result = pipeline(
    prompt="A serene Japanese garden with cherry blossoms, stone lanterns, and a koi pond",
    num_inference_steps=50,
    guidance_scale=7.5,
)

# Export mesh
result.export_mesh("garden.glb")

# Export panorama
result.save_panorama("garden_panorama.png")
```

### Image to 3D World

```python
from zen_world import ZenWorldPipeline
from PIL import Image

pipeline = ZenWorldPipeline.from_pretrained("zenlm/zen-world")
image = Image.open("scene.jpg")

result = pipeline(image=image, expand_to_360=True)
result.export_mesh("scene_3d.glb")
```

### Interactive Scene

```python
# Access individual objects
for obj in result.objects:
    print(f"{obj.label}: {obj.bbox_3d}")
    obj.export_mesh(f"{obj.label}.glb")

# Render from any viewpoint
frame = result.render(yaw=45, pitch=15, fov=90)
frame.save("view.png")
```

## Architecture

Zen World uses a three-stage pipeline:

1. **Panoramic proxy generation**: Creates a 360° panoramic representation of the scene
2. **Semantic layering**: Decomposes the scene into foreground objects and background layers
3. **Hierarchical 3D reconstruction**: Lifts the layered representation into a consistent 3D mesh

Key design properties:
- 360° immersive output via panoramic world proxies
- Standard mesh export for compatibility with existing graphics tools
- Disentangled object representations for augmented interactivity

## Lite Version

```bash
# zen-world-lite runs on consumer GPUs (16GB+ VRAM)
from zen_world import ZenWorldPipeline

pipeline = ZenWorldPipeline.from_pretrained("zenlm/zen-world-lite")
```

## Applications

- **Virtual reality**: Immersive environment generation
- **Game development**: Procedural world creation
- **Physical simulation**: 3D environment scaffolding
- **Interactive content**: Web and mobile 3D experiences
- **Film & animation**: Pre-visualization and set extension

## Links

- Models: [huggingface.co/zenlm](https://huggingface.co/zenlm)
- Docs: [zenlm.org](https://zenlm.org)

## License

Apache 2.0 — Copyright 2024 Zen LM Authors