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

#include "parameters.h"

namespace vk_gaussian_splatting {

// no reset function on purpose
SceneParameters prmScene{};

VramDataParameters    prmData{};

// no reset function on purpose
uint32_t            prmSelectedPipeline = PIPELINE_MESH;
shaderio::FrameInfo prmFrame{};
RenderParameters    prmRender{};
RasterParameters    prmRaster{};

// Storage for respective default values

static VramDataParameters    prmDataDefault{};

static shaderio::FrameInfo prmFrameDefault{};
static RenderParameters    prmRenderDefault{};
static RasterParameters    prmRasterDefault{};

void storeDefaultParameters()
{
  prmDataDefault    = prmData;

  prmFrameDefault  = prmFrame;
  prmRenderDefault = prmRender;
  prmRasterDefault = prmRaster;
}

void resetDataParameters()
{
  prmData = prmDataDefault;
}
void resetFrameParameters()
{
  prmFrame = prmFrameDefault;
}
void resetRenderParameters()
{
  prmRender = prmRenderDefault;
}
void resetRasterParameters()
{
  prmRaster = prmRasterDefault;
}

void registerCommandLineParameters(nvutils::ParameterRegistry* parameterRegistry)
{
  // Scene
  parameterRegistry->add({"inputFile", "load a ply or an spz file"}, {".ply", ".spz"}, &prmScene.sceneToLoadFilename);
#ifdef WITH_DEFAULT_SCENE_FEATURE
  parameterRegistry->add({"loadDefaultScene", "0=disable the load of a default scene when no ply file is provided"},
                         &prmScene.enableDefaultScene);
#endif
  // Projects
  parameterRegistry->add({"inputProject", "load a vkgs project file"}, {".vkgs"}, &prmScene.projectToLoadFilename);

  // Data
  parameterRegistry->add({"shformat", "0=fp32 1=fp16 2=uint8"}, &prmData.shFormat);

  // Pipelines
  parameterRegistry->add({"pipeline", "0=3dgs-vert 1=3dgs-mesh(default)"},
                         &prmSelectedPipeline);
  parameterRegistry->add({"maxShDegree", "max sh degree used for rendering in [0,1,2,3]"}, &prmRender.maxShDegree);
  parameterRegistry->add({"extentProjection", "particle extent projection method [0=Eigen (default),1=Conic]"},
                         &prmRaster.extentProjection);
}

}  // namespace vk_gaussian_splatting
