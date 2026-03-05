/*
 * Copyright (c) 2023-2025, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <nvutils/parameter_registry.hpp>

#include "shaderio.h"

namespace vk_gaussian_splatting {

// Parameters that controls the scene
struct SceneParameters
{
#ifdef WITH_DEFAULT_SCENE_FEATURE
  // do we load a default scene at startup if none is provided through CLI
  bool enableDefaultScene = true;
#endif

  // triggers a scene load at next frame when set to non empty string
  std::filesystem::path sceneToLoadFilename;
  // triggers a project load at next frame when set to non empty string
  std::filesystem::path projectToLoadFilename;
  // triggers an obj file import at next frame when set to non empty string
  std::filesystem::path meshToImportFilename;
};

// Parameters that controls the scene
extern SceneParameters prmScene;

// Parameters that controls data format and storage in VRAM, shared by all pipeline
struct VramDataParameters
{
  int shFormat    = FORMAT_FLOAT32;
  int dataStorage = STORAGE_BUFFERS;
};

// Parameters that controls data storage
extern VramDataParameters prmData;

// Parameters common to all rendering pipelines and provided to shaders as a UniformBufffer
// FrameInfo is defined in shaderio.h since declaration is shared with shaders
extern shaderio::FrameInfo prmFrame;

// pipeline selector
extern uint32_t prmSelectedPipeline;

// Parameters common to all rendering pipelines
struct RenderParameters
{
  int  visualize               = VISUALIZE_FINAL;
  bool wireframe               = false;  // display bounding volume
  int  maxShDegree             = 3;      // in [0,3]
  bool showShOnly              = false;
  bool opacityGaussianDisabled = false;
};

// Parameters common to all rendering pipelines
extern RenderParameters prmRender;

// Parameters that control rasterization
struct RasterParameters
{
  int32_t sortingMethod           = SORTING_GPU_SYNC_RADIX;
  bool    cpuLazySort             = true;  // if true, sorting starts only if viewpoint changed
  int     frustumCulling          = FRUSTUM_CULLING_AT_DIST;
  int     distShaderWorkgroupSize = 256;  // best default value set by experimentation on ADA6000
  int     meshShaderWorkgroupSize = 32;   // best default value set by experimentation on ADA6000
  bool    fragmentBarycentric     = false;
  bool    pointCloudModeEnabled   = false;
  int     extentProjection        = EXTENT_CONIC;
  // Whether gaussians should be rendered with mip-splat
  // antialiasing https://niujinshuchong.github.io/mip-splatting/
  bool msAntialiasing = false;
};

// Parameters that control rasterization
extern RasterParameters prmRaster;

// Invoked by main() to save defaults after command line options are applied at startup
void storeDefaultParameters();

// Reset prmData to defaults
void resetDataParameters();

// Reset prmFrame to defaults
void resetFrameParameters();
// Reset prmRender to defaults
void resetRenderParameters();
// Reset prmRaster to defaults
void resetRasterParameters();

// register the set of global parameters
void registerCommandLineParameters(nvutils::ParameterRegistry* parameterRegistry);

}  // namespace vk_gaussian_splatting
