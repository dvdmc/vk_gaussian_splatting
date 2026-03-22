# Implementation Notes: vk_gaussian_splatting + GLTF

This document explains how the renderer is structured, why certain decisions were made,
and introduces the underlying graphics concepts for readers with little prior knowledge.

---

## Table of Contents

**Architecture reference**
1. [What this renderer does](#1-what-this-renderer-does)
2. [How Vulkan pipelines work — the basics](#2-how-vulkan-pipelines-work--the-basics)
3. [The two rendering systems and how they differ](#3-the-two-rendering-systems-and-how-they-differ)
4. [The GLTF render pipeline — GltfRasterizer](#4-the-gltf-render-pipeline--gltfrasterizer)
5. [VkShaderEXT — what it is and why it was chosen](#5-vkshaderext--what-it-is-and-why-it-was-chosen)
6. [Environment lighting — sky and IBL](#6-environment-lighting--sky-and-ibl)
7. [The frame — what happens every render call](#7-the-frame--what-happens-every-render-call)
8. [Image layouts — what they are and why they matter](#8-image-layouts--what-they-are-and-why-they-matter)
9. [How the two systems share the framebuffer](#9-how-the-two-systems-share-the-framebuffer)
10. [Known issues in the integration](#10-known-issues-in-the-integration)
11. [Third-party libraries — tiny_stb_implementation.cpp](#11-third-party-libraries--tiny_stb_implementationcpp)
12. [File map](#12-file-map)

**Multi-camera implementation plan**

13. [Design decisions](#13-design-decisions)
14. [Implementation plan](#14-implementation-plan)

---

## 1. What this renderer does

This application renders two kinds of content together in the same image:

- **Gaussian splats** — a scene representation where objects are described as millions of
  small semi-transparent blobs (Gaussians) rather than triangles. They are sorted back to
  front every frame and blended together to form a photograph-like image.

- **GLTF meshes** — standard 3D models in the industry-standard GLTF format, rendered with
  physically-based materials (PBR): metallic surfaces, rough/smooth materials, textures, etc.

The two are rendered sequentially into the same color and depth image so that GLTF geometry
and gaussian splats can coexist in the same scene, with correct depth relationships between them.

---

## 2. How Vulkan pipelines work — the basics

Vulkan is a low-level graphics API. To draw anything you need to tell the GPU:

- **What shaders to run** — programs written in GLSL or Slang that run on the GPU.
  - A *vertex shader* transforms 3D positions to screen positions.
  - A *fragment shader* decides the color of each pixel.
  - A *mesh shader* is a modern replacement for the vertex shader with more flexibility.
  - A *compute shader* does arbitrary GPU computation (sorting, culling, etc.)

- **What state to use** — blending mode, depth testing, culling mode, viewport size, etc.

In traditional Vulkan, all of this is compiled together into a `VkPipeline` object —
a single immutable object baked at startup. Changing any piece of state means
creating a new pipeline object. This is predictable and fast at draw time, but inflexible.

A newer extension, `VK_EXT_shader_object`, introduces `VkShaderEXT` — shader objects that
can be bound independently, with all state set dynamically on the command buffer at draw time.
This is more flexible but requires more command buffer work per draw call.

This renderer uses **both approaches** for different parts of the frame.

---

## 3. The two rendering systems and how they differ

### Gaussian Splatting — traditional `VkPipeline`

Defined in: `src/gaussian_splatting.cpp`, `initPipelines()`

Five pipeline objects are created at startup:

| Pipeline object | Purpose |
|---|---|
| `m_computePipelineGsDistCull` | Compute shader: per-splat distance calculation and frustum culling |
| `m_graphicsPipelineGsMesh` | Mesh shader path for 3DGS gaussian splats |
| `m_graphicsPipeline3dgutMesh` | Mesh shader path for 3DGUT gaussian splats |
| `m_graphicsPipelineGsVert` | Vertex shader fallback for 3DGS splats |
| `m_graphicsPipelineMesh` | Legacy rasterization pipeline (retained for non-GLTF meshes, NOW REMOVED) |

All blend equations, cull modes, and vertex input formats are **baked in** at creation time.
The only dynamic state used is `VK_DYNAMIC_STATE_DEPTH_WRITE_ENABLE` and
`VK_DYNAMIC_STATE_DEPTH_TEST_ENABLE`, toggled depending on whether alpha sorting is active.

At draw time: `vkCmdBindPipeline` → `vkCmdBindDescriptorSets` → draw.

### GLTF Rasterizer — `VkShaderEXT`

Defined in: `src/gltf_rasterizer.cpp`, `src/gltf_rasterizer.h`

Three `VkShaderEXT` objects are created from a single SPIR-V binary (`gltf_raster.slang`),
each selecting a different entry point:

| Shader object | Entry point | Purpose |
|---|---|---|
| `m_vertexShader` | `vertexMain` | Transform vertices to screen space |
| `m_fragmentShader` | `fragmentMain` | Full PBR shading |
| `m_wireframeShader` | `fragmentWireframeMain` | Debug wireframe mode |

All pipeline state (cull mode, blend, depth bias, vertex input) is set dynamically on the
command buffer each frame. This allows the GLTF renderer to change state between material
batches — solid geometry, double-sided geometry, and alpha-blended geometry — without
maintaining separate pipeline objects for each.

### Why different approaches?

The splat pipelines have a **fixed, known set of variants** (3DGS vs 3DGUT, mesh vs vertex
shader). There is no per-frame state switching. `VkPipeline` is optimal here.

The GLTF renderer must vary cull mode, depth bias, and blending **per material batch within
the same frame** depending on what each GLTF material needs. `VkShaderEXT` with dynamic state
is the right fit — no need to pre-enumerate all material permutations.

### Comparison table

| | Gaussian Splatting (`VkPipeline`) | GLTF (`VkShaderEXT`) |
|---|---|---|
| State baking | At startup | At draw time via `vkCmdSet*` |
| Per-frame state changes | Minimal (depth test only) | Cull mode, blend, depth bias |
| Shader variants | One pipeline per variant | One shader object, state changes between batches |
| Vertex input | Baked into pipeline | `vkCmdSetVertexInputEXT` |
| Descriptor set | Splat buffers and textures | Bindless scene textures + IBL cubemaps |
| Pipeline layout | Separate | Separate |

The two systems have separate descriptor set layouts and pipeline layouts. They are
**independent** — neither knows about the internals of the other.

---

## 4. The GLTF render pipeline — GltfRasterizer

`GltfRasterizer` is a self-contained class that owns everything needed to render a GLTF scene:

```
GltfRasterizer
├── nvvkgltf::Scene        — GLTF scene in CPU memory (nodes, materials, primitives)
├── nvvkgltf::SceneVk      — GPU buffers: vertices, indices, uploaded textures
├── nvvk::HdrIbl           — HDR environment image + pre-filtered cubemaps
├── nvshaders::HdrEnvDome  — Renders the HDR image as a background dome
├── nvshaders::SkyPhysical — Compute-shader physical sky (fallback environment)
├── m_frameInfoBuffer      — Per-frame GPU buffer: view/proj matrices, env settings
├── m_skyParamsBuffer      — Per-frame GPU buffer: sun direction, intensity, etc.
└── m_descSet              — Descriptor set: bindless textures, cube maps, HDR/LUT
```

### Descriptor bindings

The descriptor set has three bindings, all using `UPDATE_AFTER_BIND` and `PARTIALLY_BOUND`
flags (meaning textures can be updated while the GPU is running, and not all slots need to
be filled):

| Binding | Type | Count | Contents |
|---|---|---|---|
| 0 (`GLTF_BINDING_TEXTURES`) | Combined image sampler | 512 | All scene textures (bindless array) |
| 1 (`GLTF_BINDING_TEXTURES_CUBE`) | Combined image sampler | 2 | Diffuse + glossy IBL cubemaps |
| 2 (`GLTF_BINDING_TEXTURES_HDR`) | Combined image sampler | 2 | Raw HDR image + BRDF LUT |

**Bindless textures** (binding 0) means all 512 texture slots are declared upfront, and the
shader indexes into them using a material ID loaded from a buffer. This avoids re-binding
descriptor sets per draw call.

### Push constants and Buffer Device Addresses

Rather than updating descriptor sets every frame, `GltfRasterPushConstant` carries
**Buffer Device Addresses** (BDA) — raw GPU pointers to the frame info and sky params buffers:

```cpp
struct GltfRasterPushConstant {
    int32_t materialID;      // which material this node uses
    int32_t renderNodeID;    // which node in the scene
    int32_t renderPrimID;    // which primitive (mesh)
    float   _pad;
    GltfSceneFrameInfo*      frameInfo;   // GPU pointer — updated per frame
    SkyPhysicalParameters*   skyParams;   // GPU pointer — updated per frame
    GltfScene*               gltfScene;  // GPU pointer — stable after load
};
```

The BDA pointers (frameInfo, skyParams, gltfScene) are set once per draw call.
The per-node fields (materialID, renderNodeID, renderPrimID) are updated for each node
via a partial push constant update using an offset, avoiding a full push constant rewrite
for each node.

### Shader: gltf_raster.slang

A Slang (HLSL-compatible) PBR shader with two main entry points:

**Vertex shader (`vertexMain`):**
- Reads vertex position from the bound vertex buffer
- Reads the node's world transform from the scene description buffer (via BDA)
- Outputs world-space position (for lighting) and clip-space position

**Fragment shader (`fragmentMain`):**
- Fetches the material from the scene description buffer
- Samples base color, normal map, metallic/roughness, emissive textures
- Evaluates PBR lighting in one of two modes:
  - `eSky` — ambient + directional sun light from physical sky parameters
  - `eHdr` — full IBL: `getIBLRadianceGGX()` for specular, `getIBLRadianceLambertian()` for diffuse
- Supports alpha masking and alpha blending
- Debug visualization mode: can display individual material properties (normals, roughness, etc.)

---

## 5. VkShaderEXT — what it is and why it was chosen

`VkShaderEXT` (from extension `VK_EXT_shader_object`) is a modern Vulkan feature that
separates shader objects from pipeline state. Instead of one monolithic `VkPipeline`, you:

1. Create lightweight `VkShaderEXT` objects from SPIR-V (like `VkShaderModule`, but bindable directly)
2. Set all state dynamically on the command buffer with `vkCmdSet*` calls
3. Bind shaders with `vkCmdBindShadersEXT`

**Creation** (`createShaders`, `gltf_rasterizer.cpp`):

All three shader objects are created from the same SPIR-V blob but with different entry points.
The `nextStage` field chains vertex → fragment so the driver knows the pipeline topology:

```cpp
VkShaderCreateInfoEXT info{
    .stage     = VK_SHADER_STAGE_VERTEX_BIT,
    .nextStage = VK_SHADER_STAGE_FRAGMENT_BIT,  // vertex feeds fragment
    .codeType  = VK_SHADER_CODE_TYPE_SPIRV_EXT,
    .pCode     = gltf_raster_slang,             // embedded SPIR-V
    .pName     = "vertexMain",                  // entry point
    ...
};
vkCreateShadersEXT(device, 1, &info, nullptr, &m_vertexShader);
```

**Usage at draw time** (`draw`, `gltf_rasterizer.cpp`):

```cpp
m_dynamicPipeline.cmdApplyAllStates(cmd);         // blend, rasterization defaults
m_dynamicPipeline.cmdSetViewportAndScissor(cmd, size);
m_dynamicPipeline.cmdBindShaders(cmd, {.vertex = m_vertexShader, .fragment = m_fragmentShader});
vkCmdSetVertexInputEXT(cmd, ...);                 // vertex format
vkCmdSetDepthTestEnable(cmd, VK_TRUE);

// Solid objects
vkCmdSetCullMode(cmd, VK_CULL_MODE_BACK_BIT);
vkCmdSetDepthBias(cmd, -1.0f, 0.0f, 1.0f);
renderNodes(cmd, solidNodes);

// Double-sided (just change cull mode — no new pipeline)
vkCmdSetCullMode(cmd, VK_CULL_MODE_NONE);
renderNodes(cmd, doubleSidedNodes);

// Alpha-blended (flip on blending — no new pipeline)
VkBool32 blendEnable = VK_TRUE;
vkCmdSetColorBlendEnableEXT(cmd, 0, 1, &blendEnable);
renderNodes(cmd, blendNodes);
```

With `VkPipeline`, those three material batches would each require a separately compiled pipeline.

**Important:** `vkCmdBindPipeline` and `vkCmdBindShadersEXT` are mutually exclusive per draw.
The two systems (splats and GLTF) must therefore render in separate `vkCmdBeginRendering` blocks,
which they already do.

---

## 6. Environment lighting — sky and IBL

The GLTF renderer supports two background/lighting modes:

### Physical sky (`eSky`)

A compute shader (`sky_physical.slang`) generates a gradient sky based on physical parameters:
sun direction, sun intensity, atmospheric scattering coefficients, etc.
It runs while the color image is in `GENERAL` layout (required for compute writes).
The resulting sky also feeds the fragment shader as a directional light source.

### HDR environment (`eHdr`)

A more realistic option using a real HDR photograph of an environment (`.hdr` file):

1. **Load**: The raw HDR image is uploaded to the GPU.
2. **Pre-filter**: Three GPU compute passes transform the image:
   - `hdr_prefilter_diffuse.slang` → diffuse irradiance cubemap (how light scatters from rough surfaces)
   - `hdr_prefilter_glossy.slang` → specular reflection cubemap with mip levels (blurrier mips = rougher reflections)
   - `hdr_integrate_brdf.slang` → BRDF lookup table (a 2D texture encoding how specular highlights behave)
3. **Render background**: `hdr_dome.slang` draws the HDR image as a background.
4. **Lighting in shader**: The fragment shader samples the pre-filtered cubemaps to compute
   IBL (Image-Based Lighting) — realistic environment reflections and ambient lighting.

**Why pre-filter?**
Sampling a raw HDR for every surface point at every angle would be too slow.
The pre-filtered cubemaps encode the result of integrating the entire environment over a
hemisphere of directions, which can be looked up with a single texture sample.
This is a standard offline-to-realtime technique called **split-sum approximation**.

The BRDF LUT encodes the specular contribution as a function of roughness and view angle,
separating it from the environment so the two can be multiplied together cheaply at runtime.

---

## 7. The frame — what happens every render call

`GaussianSplatting::onRender(cmd)` runs once per frame. Here is the full sequence:

```
1. processUpdateRequests()
   — If shaders/data changed (e.g. new splat file loaded), rebuild pipelines now.

2. Splat sorting (if splats loaded)
   — updateAndUploadFrameInfoUBO(): upload camera matrices, splat count, etc.
   — GPU path: processSortingOnGPU() — radix sort on GPU, fully async.
   — CPU path: tryConsumeAndUploadCpuSortingResult() — sorted indices from CPU thread.

3. depth image: UNDEFINED → GENERAL
   — Discard last frame's depth (UNDEFINED src layout = don't care about old content).
   — GENERAL layout is required for the sky compute shader to write to it.

4. GltfRasterizer::draw()
   — Uploads frame info buffer (view/proj, env settings).
   — Runs sky compute or HDR dome (writes background into color image, GENERAL layout).
   — Transitions color + depth: GENERAL → COLOR/DEPTH_ATTACHMENT_OPTIMAL.
   — Begins vkCmdBeginRendering, draws GLTF nodes (solid → double-sided → blended).
   — Ends rendering.
   — Transitions color + depth back: ATTACHMENT → GENERAL.

5. color + depth: GENERAL → COLOR/DEPTH_ATTACHMENT_OPTIMAL
   — Prepare for the splat rendering pass.

6. vkCmdBeginRendering (splat pass)
   — colorAttachment.loadOp = LOAD  (always — GLTF/sky already rendered)
   — depthAttachment.loadOp = LOAD  (only if GLTF scene present — respect GLTF occlusion)
                            = CLEAR (no GLTF scene — fresh depth for splats only)
   — drawSplatPrimitives(): bind pipeline, draw splats back-to-front, alpha-blend over GLTF.
   — vkCmdEndRendering.

7. color + depth: ATTACHMENT → GENERAL
   — Return images to the framework for display/compositing.
```

### Why GLTF renders first

Gaussian splats are semi-transparent and sorted back-to-front. They must be blended
*over* the GLTF geometry using `LOAD_OP_LOAD`. If splats rendered first, the GLTF
geometry would overwrite them.

GLTF geometry writes to the depth buffer, so when the splat pass loads that depth,
splats behind GLTF objects are naturally occluded. This is why depth is loaded with
`LOAD_OP_LOAD` when a GLTF scene is present.

---

## 8. Image layouts — what they are and why they matter

Vulkan images are not just memory — they have a **layout** that tells the GPU how the
data is arranged internally (tiled for rendering, linear for transfer, etc.).

The GPU can only use an image correctly if it is in the right layout for the operation.
Transitioning between layouts is done with **image memory barriers** (`vkCmdPipelineBarrier`
or `vkCmdImageMemoryBarrier`), which also act as synchronization points.

Layouts used in this renderer:

| Layout | Used for |
|---|---|
| `UNDEFINED` | "I don't care about the current content" — used as source when discarding |
| `GENERAL` | Compute shader reads/writes; also a universal fallback (slower for raster) |
| `COLOR_ATTACHMENT_OPTIMAL` | Fragment shader color output — fastest for rasterization |
| `DEPTH_ATTACHMENT_OPTIMAL` | Depth buffer during rasterization |
| `TRANSFER_SRC_OPTIMAL` | Copying image data to a buffer (readback) |

**Why `UNDEFINED` for depth at the start of the frame?**
Using `UNDEFINED` as the source layout tells the driver it can discard the previous
content without reading it. This avoids unnecessary memory traffic, which is a
measurable performance win on tile-based GPUs (mobile) and a free optimization
on desktop GPUs.

---

## 9. How the two systems share the framebuffer

The two rendering passes share the same color and depth images, passing them
back and forth via layout transitions. Neither system owns the images — they are
owned by `nvvk::GBuffer` and passed by handle to each pass.

```
┌─────────────────────────────────────────────────────────┐
│  GBuffer (owned by GaussianSplatting)                   │
│  color image    depth image                             │
└────────┬──────────────┬────────────────────────────────┘
         │              │
         ▼              ▼
  [UNDEFINED→GENERAL for depth; color assumed GENERAL from prev frame]
         │              │
         ▼              ▼
  ┌──────────────────────────┐
  │   GltfRasterizer::draw() │  ← sky compute, GLTF raster
  │   enters: GENERAL        │
  │   exits:  GENERAL        │
  └──────────────────────────┘
         │              │
         ▼              ▼
  [GENERAL → ATTACHMENT for both]
         │              │
         ▼              ▼
  ┌──────────────────────────┐
  │   Splat pass             │  ← LOAD_OP_LOAD, blend over GLTF
  │   enters: ATTACHMENT     │
  │   exits:  GENERAL        │
  └──────────────────────────┘
         │              │
         ▼              ▼
     displayed / next frame
```

The systems are **independent** — separate descriptor sets, pipeline layouts, and shader
objects. They communicate only via the shared framebuffer images and their layout contracts
(what layout is expected on entry, what is guaranteed on exit).

---

## 10. Known issues in the integration

### Color image lacks an explicit initial layout transition

The depth image is explicitly transitioned `UNDEFINED → GENERAL` at the start of every frame.
The color image is not — `GltfRasterizer::draw()` enters expecting it in `GENERAL` layout,
relying on `GBuffer` to have initialized it to `GENERAL` at creation time, and on the previous
frame having left it in `GENERAL` on exit.

On frame 0, if `GBuffer` does not perform an initial `UNDEFINED → GENERAL` transition for
the color image, the Vulkan validation layer will report an invalid source layout error.
The fix is to add the same explicit barrier for the color image that already exists for depth.

### Sorting runs once per GBuffer

The splat sorting and frame info upload code lives inside `for(nvvk::GBuffer& gBuffer : m_gBuffers)`.
If `m_gBuffers` has more than one entry (e.g. for stereo or multi-view rendering), the sort
runs redundantly for each buffer. The sort output is identical each iteration, so this is
not incorrect — just wasteful. The sorting should ideally move outside the loop.

### Unused clear value

```cpp
colorAttachment.loadOp     = VK_ATTACHMENT_LOAD_OP_LOAD;  // always LOAD
colorAttachment.clearValue = {m_clearColor};               // never used
```

When `loadOp` is `LOAD`, `clearValue` is ignored by the driver. The line is harmless
but misleading — the actual background clear happens inside `GltfRasterizer` (either
via the sky compute, the HDR dome, or a `CLEAR` load op for solid background mode).

---

## 11. Third-party libraries — tiny_stb_implementation.cpp

`tinygltf` and `stb_image` are **header-only libraries**: a single `.h` file contains
both declarations and full function implementations, guarded by a macro:

```cpp
// Inside tiny_gltf.h (simplified):
#ifdef TINYGLTF_IMPLEMENTATION
  // actual function bodies, thousands of lines
#endif
```

If you include the header without the macro, you only get declarations — the linker
will fail because the symbols were never emitted. If multiple `.cpp` files define the
macro, the linker fails with duplicate symbols.

The rule is: **define the macro in exactly one translation unit**. That is the sole purpose
of `tiny_stb_implementation.cpp` — it is a dedicated file that does nothing except define
those macros and include the headers.

```cpp
#define TINYGLTF_IMPLEMENTATION       // emit tinygltf function bodies
#define STB_IMAGE_IMPLEMENTATION      // emit stb_image function bodies
#define STB_IMAGE_WRITE_IMPLEMENTATION
#define TINYGLTF_NO_EXTERNAL_IMAGE    // suppress tinygltf's built-in image loading
                                      // (nvvkgltf::SceneVk handles GPU texture upload)
#include <tinygltf/tiny_gltf.h>
```

The `#pragma warning` blocks suppress signed/unsigned and size_t truncation warnings
from third-party code that you do not control and should not modify.

---

## 12. File map

```
src/
├── gaussian_splatting.h/.cpp     — Main renderer class: splat pipelines, onRender, sorting
├── gaussian_splatting_ui.h/.cpp  — ImGui UI panels
├── gltf_rasterizer.h/.cpp        — Self-contained GLTF + IBL renderer (VkShaderEXT)
├── splat_set_vk.h/.cpp           — GPU buffers for gaussian splat data
├── splat_sorter_async.h/.cpp     — Background CPU thread for splat sorting
├── ply_loader_async.h/.cpp       — Background thread for .ply file loading
├── mesh_set_vk.h/.cpp            — GPU buffers for legacy OBJ mesh data
├── light_set_vk.h/.cpp           — GPU buffer for scene lights
├── parameters.h/.cpp             — Serializable render parameters
├── tiny_stb_implementation.cpp   — Single-TU macro definitions for tinygltf + stb_image
└── main.cpp                      — Application entry point, Vulkan instance setup

shaders/
├── gltf_raster.slang             — PBR vertex + fragment shader for GLTF
└── gltf_shaderio.h               — Shared CPU/GPU structs (push constants, frame info, etc.)
```

The reference implementation from NVIDIA (`vk_gltf_renderer/`) lives alongside this project
and was the source for the IBL pre-filtering shaders and `GltfRasterizer` design.

---

## 13. Design decisions

This section records the reasoning behind architectural choices made for the multi-camera
interface and its future integration with a ROS node.

### Should GaussianSplatting be decoupled from nvapp::Application?

**Decision: No. Keep the app wrapper.**

`nvapp::Application` already supports headless mode (`appInfo.headless = true`). In that
mode there is no window or swapchain, but the render loop, command buffer recording, queue
submission, and frame-in-flight synchronization all continue to work. Removing the app
wrapper would require reimplementing all of that manually.

The class's actual dependencies on `m_app` are narrow:
- `m_app->getQueue(0)` — for the transient command pool
- `m_app->getViewportSize()` — for the splat viewport

These are not reasons to strip the wrapper. For a headless ROS mode, set
`appInfo.headless = true` in `main.cpp` and remove the window-specific elements.

### GUI + ROS on different threads — is it a problem?

**Decision: Not a problem. Standard producer-consumer.**

`nvapp::Application::run()` — and all Vulkan command recording and ImGui rendering —
lives on a single thread. ROS nodes spin on their own threads. The two never touch Vulkan
simultaneously.

The only data crossing the thread boundary:
- **Camera poses in**: a ROS subscriber writes a pose; `onRender` reads it.
  A `std::mutex` around `m_cameraSet` writes is sufficient.
- **Images out**: after `onRender` writes pixels into `m_cameraReadbacks[i].image`,
  a ROS publisher thread reads that buffer.
  A per-camera mutex + a "new frame ready" flag covers this.

No Vulkan objects are touched from the ROS thread. This pattern is safe and straightforward.

### How to render multiple cameras

**Decision: sequential re-sort + re-render per camera, one shared offscreen GBuffer.**

Gaussian splat sorting is view-dependent: splats must be ordered back-to-front from
each camera's eye position. Rendering N cameras correctly therefore requires N sorts.

**Why not reuse one sort?**
Using the interactive camera's sort order for virtual cameras produces incorrect blending
when the virtual cameras are at significantly different positions. For up to 5 cameras
spread around a scene this would be visibly wrong.

**Why not one GBuffer per camera?**
Memory scales with N and the display path becomes more complex. A single reused offscreen
GBuffer (render → readback → render next camera) is simpler and uses constant VRAM.

**Why only GPU sort for virtual cameras?**
The CPU sorter (`SplatSorterAsync`) runs one background sort per frame and delivers the
result on the next frame. For N virtual cameras you would need N background sorters and
N frames of latency. The GPU sorter (`VrdxSorter`) is synchronous within the command
buffer and can be called N times per frame. For ≤5 cameras at typical splat counts this
is fast enough. When virtual cameras are active, the rendering path forces GPU sort
regardless of the user's sort setting.

**Rendering order:**
```
For each virtual camera i (indices 1..N-1, or 0..N-1 when headless):
  1. Load camera i matrices (eye, view, proj) from m_cameraSet.getPreset(i)
  2. Upload frame info UBO for camera i
  3. processSortingOnGPU (distance compute + radix sort for this eye)
  4. GltfRasterizer::draw (sky/HDR + GLTF geometry)
  5. Splat pass (LOAD_OP_LOAD, blend over GLTF)
  6. Transition color → TRANSFER_SRC_OPTIMAL
  7. vkCmdCopyImageToBuffer → m_cameraReadbacks[i].buffer
  8. Transition color → GENERAL for next camera iteration
  9. Lock m_cameraReadbacks[i].mutex, memcpy to .image, set .ready = true, unlock
```

The display camera (index 0, driven by `cameraManip`) continues to render into the
existing `m_gBuffers` double-buffer loop, unchanged. Virtual cameras use a separate
`m_offscreenGBuffer` that is never composited to the swapchain.

### Camera 0 — interactive vs. virtual

When the application is **not headless**, camera 0 is the interactive display camera.
It is driven by `cameraManip` (mouse/keyboard navigation) and rendered into `m_gBuffers`.
Its image is displayed in the application window.

When the application **is headless**, there is no user interaction. Camera 0 is treated
as a virtual camera like all others.

The `CameraSet` already stores camera presets. The convention adopted here:
- `m_cameraSet.getPreset(0)` = the home position (starting pose for the interactive camera)
- `m_cameraSet.getPreset(i)` for i ≥ 1 = virtual cameras with fixed poses

---

## 14. Implementation plan

### Step 1 — Replace cameraPositions/activeCamera with CameraSet

**Files:** `gaussian_splatting.h`, `gaussian_splatting.cpp`

Remove the ad-hoc `cameraPositions` vector and `activeCamera` int that were added as
placeholders. The `CameraSet m_cameraSet` already exists and already holds camera presets.

Changes:
- Delete `std::vector<glm::vec3> cameraPositions` and `int activeCamera` from the header.
- In `updateAndUploadFrameInfoUBO`, remove the `cameraManip->setEye(cameraPositions[activeCamera])`
  hack. The display camera always reads from `cameraManip` directly (no change to the rest
  of the UBO upload logic).
- Expose the camera management interface on `GaussianSplatting` (see Step 5).

### Step 2 — Per-camera readback buffers

**Files:** `gaussian_splatting.h`, `gaussian_splatting.cpp`

Add a struct and a vector to hold one readback buffer and one CPU image per virtual camera:

```cpp
struct CameraCapture {
    nvvk::Buffer               buffer;   // GPU → CPU readback buffer
    std::vector<unsigned char> image;    // CPU-side pixels after readback
    bool                       ready = false;  // true when image is valid for this frame
    std::mutex                 mutex;    // protects image and ready for ROS thread access
};
std::vector<CameraCapture> m_cameraCaptures;  // one entry per virtual camera (index 1..N-1)
```

`m_cameraCaptures` is resized when cameras are added or removed (via `addCamera` /
`removeCamera`). Each buffer is sized to `width × height × 4` bytes (RGBA8).

The existing `m_readbackBuffer` and `image` on the class remain for the display camera
(camera 0) readback, which is used by the existing benchmark/screenshot path.

### Step 3 — Offscreen GBuffer for virtual cameras

**Files:** `gaussian_splatting.h`, `gaussian_splatting.cpp`

Add a single offscreen GBuffer used by all virtual cameras sequentially:

```cpp
nvvk::GBuffer m_offscreenGBuffer;          // reused across virtual cameras each frame
VkExtent2D    m_virtualCameraResolution;   // configurable, defaults to display resolution
```

Init/deinit alongside the existing `m_gBuffers`. Resize when `setVirtualCameraResolution`
is called or when the display resizes (if resolution is set to follow display).

The offscreen GBuffer has the same format as the display GBuffer (`m_colorFormat`,
`m_depthFormat`) and is never blitted to the swapchain.

### Step 4 — Multi-camera loop in onRender

**Files:** `gaussian_splatting.cpp`

The existing display camera path inside `for(nvvk::GBuffer& gBuffer : m_gBuffers)` is
**not modified**. After that loop, add the virtual camera pass:

```
if (m_cameraCaptures is not empty)
{
    ensure GPU sort is used for virtual cameras (warn/override if CPU sort is selected)

    for i = 1 .. m_cameraSet.size() - 1:
        camera = m_cameraSet.getPreset(i)
        set view/proj matrices from camera (do NOT touch cameraManip)
        uploadFrameInfoUBO for camera i (write into m_frameInfoBuffer)
        processSortingOnGPU (distance compute + radix sort)

        depth: UNDEFINED → GENERAL
        m_gltfRasterizer.draw(cmd, m_offscreenGBuffer, view, proj)
        color/depth: GENERAL → ATTACHMENT
        vkCmdBeginRendering (LOAD_OP_LOAD color, LOAD_OP_LOAD depth if GLTF scene)
        drawSplatPrimitives(cmd, splatCount)
        vkCmdEndRendering

        color: ATTACHMENT → TRANSFER_SRC_OPTIMAL
        vkCmdCopyImageToBuffer → m_cameraCaptures[i-1].buffer
        pipeline barrier (TRANSFER_WRITE → HOST_READ)
        color: TRANSFER_SRC_OPTIMAL → GENERAL   // ready for next iteration

    after all cameras: memcpy from mapped readback buffers → .image, set .ready = true
    (or do this at the start of the next frame after the GPU has finished)
}
```

Note: `processSortingOnGPU` writes to `m_splatDistancesDevice` and `m_splatIndicesDevice`,
which are also used by the display camera. The virtual camera loop runs after the display
camera's `vkCmdEndRendering`, so there is no overlap — the sort buffers can be reused safely.

### Step 5 — Public camera interface

**Files:** `gaussian_splatting.h`, `gaussian_splatting.cpp`

Add the following public methods to `GaussianSplatting`. These are safe to call from
the application thread at any time (the next `onRender` call will pick up the changes).
For ROS thread calls, protect with the mutex described in Step 2.

```cpp
// Add a virtual camera. Returns its index in m_cameraSet (always ≥ 1).
// Allocates a readback buffer and extends m_cameraCaptures.
uint64_t addCamera(const Camera& cam);

// Remove virtual camera at index. Index 0 (display camera) cannot be removed.
// Frees the readback buffer. Returns false if index is invalid.
bool removeCamera(uint64_t index);

// Replace the pose of virtual camera at index. Returns false if index is invalid.
bool setCamera(uint64_t index, const Camera& cam);

// Read the pose of camera at index.
Camera getCamera(uint64_t index) const;

// Number of cameras including the display camera (index 0).
uint64_t getCameraCount() const;

// Copy the latest rendered image of virtual camera at index into `out`.
// Thread-safe: acquires the per-camera mutex.
// Returns false if no image is ready yet or index is out of range.
bool getImage(uint64_t index, std::vector<unsigned char>& out);

// Resolution used for all virtual camera renders.
// Changing this destroys and recreates m_offscreenGBuffer and all readback buffers.
void setVirtualCameraResolution(VkExtent2D size);
VkExtent2D getVirtualCameraResolution() const;
```

`getImage` is the ROS-safe entrypoint: the publisher thread calls it every time it wants
a new image, and the mutex ensures it never reads a partially written buffer.

### Step 6 — CPU sort deprecation for multi-camera

**Files:** `gaussian_splatting.cpp`, `gaussian_splatting_ui.cpp` (UI warning)

CPU sort (`SplatSorterAsync`) is inherently single-camera: it runs one async sort per frame
and cannot be run N times in parallel in a single frame without N separate sorter instances
and N frames of latency.

For now, rather than removing the CPU sort, add a guard in `onRender`:

```cpp
if (!m_cameraCaptures.empty() && prmRaster.sortingMethod != SORTING_GPU_SYNC_RADIX)
{
    LOGW("Multi-camera rendering requires GPU sort. Overriding sorting method.\n");
    // use GPU sort path for virtual camera loop
}
```

The UI can show a warning or disable the CPU sort option when virtual cameras are present.
Full removal of the CPU sorter is deferred until the multi-camera path is validated.

### Summary table

| Step | What changes | Files touched |
|---|---|---|
| 1 | Remove cameraPositions/activeCamera; wire CameraSet | gaussian_splatting.h/.cpp |
| 2 | Per-camera readback buffers + CameraCapture struct | gaussian_splatting.h/.cpp |
| 3 | Offscreen GBuffer + configurable virtual resolution | gaussian_splatting.h/.cpp |
| 4 | Virtual camera loop in onRender | gaussian_splatting.cpp |
| 5 | Public camera interface (add/remove/set/get/getImage) | gaussian_splatting.h/.cpp |
| 6 | CPU sort guard for multi-camera | gaussian_splatting.cpp, _ui.cpp |
