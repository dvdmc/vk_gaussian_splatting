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

#ifndef GLTF_SHADERIO_H
#define GLTF_SHADERIO_H

#include "nvshaders/slang_types.h"
#include "nvshaders/sky_io.h.slang"
#include "nvshaders/gltf_scene_io.h.slang"

NAMESPACE_SHADERIO_BEGIN()

#define HDR_DIFFUSE_INDEX 0
#define HDR_GLOSSY_INDEX  1
#define HDR_IMAGE_INDEX   0
#define HDR_LUT_INDEX     1

// Binding points for GLTF descriptor set (set 0)
#define GLTF_BINDING_TEXTURES      0
#define GLTF_BINDING_TEXTURES_CUBE 1
#define GLTF_BINDING_TEXTURES_HDR  2

// Environment types
enum class EnvSystem
{
  eSky,
  eHdr,
};

// Debug visualization methods
enum DebugMethod
{
  eNone,
  eBaseColor,
  eMetallic,
  eRoughness,
  eNormal,
  eTangent,
  eBitangent,
  eEmissive,
  eOpacity,
  eTexCoord0,
  eTexCoord1,
};

// Per-frame camera and environment info passed via buffer device address
struct GltfSceneFrameInfo
{
  float4x4    viewMatrix;
  float4x4    projInv;
  float4x4    viewInv;
  float4x4    viewProjMatrix;
  int         isOrthographic    SLANG_DEFAULT(0);
  float       envRotation       SLANG_DEFAULT(0.f);
  float       envBlur           SLANG_DEFAULT(0.f);
  float       envIntensity      SLANG_DEFAULT(1.f);
  int         useSolidBackground SLANG_DEFAULT(0);
  float3      backgroundColor   SLANG_DEFAULT(float3(0, 0, 0));
  int         environmentType   SLANG_DEFAULT(0);
  DebugMethod debugMethod       SLANG_DEFAULT(DebugMethod::eNone);
};

// Push constant for GLTF rasterization
struct GltfRasterPushConstant
{
  int                    materialID   SLANG_DEFAULT(0);
  int                    renderNodeID SLANG_DEFAULT(0);
  int                    renderPrimID SLANG_DEFAULT(0);
  float                  _pad         SLANG_DEFAULT(0.f);
  GltfSceneFrameInfo*    frameInfo;
  SkyPhysicalParameters* skyParams;
  GltfScene*             gltfScene;
};

NAMESPACE_SHADERIO_END()

#endif  // GLTF_SHADERIO_H
