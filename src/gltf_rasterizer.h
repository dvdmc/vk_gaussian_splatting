/*
 * Copyright (c) 2023-2026, NVIDIA CORPORATION.  All rights reserved.
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
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <filesystem>
#include <memory>

#include <glm/glm.hpp>
#include <vulkan/vulkan_core.h>

#include <nvapp/application.hpp>
#include <nvvk/descriptors.hpp>
#include <nvvk/gbuffers.hpp>
#include <nvvk/graphics_pipeline.hpp>
#include <nvvk/hdr_ibl.hpp>
#include <nvvk/resource_allocator.hpp>
#include <nvvk/sampler_pool.hpp>
#include <nvvk/staging.hpp>

#include <nvvkgltf/scene.hpp>
#include <nvvkgltf/scene_vk.hpp>

#include <nvshaders_host/hdr_env_dome.hpp>
#include <nvshaders_host/sky.hpp>

#include "gltf_shaderio.h"

namespace vk_gaussian_splatting {

// GltfRasterizer handles GLTF scene loading, IBL environment, and rasterization
// using VkShaderEXT (VK_EXT_shader_object) for dynamic pipeline state.
class GltfRasterizer
{
public:
  GltfRasterizer()  = default;
  ~GltfRasterizer() = default;

  // Initialize with application context. Call once at startup.
  void init(nvapp::Application*      app,
            nvvk::ResourceAllocator* alloc,
            nvvk::SamplerPool*       samplerPool,
            VkFormat                 colorFormat,
            VkFormat                 depthFormat);

  void deinit();

  // Load a GLTF/GLB scene file. Returns true on success.
  bool loadGltfScene(const std::filesystem::path& path);
  void unloadGltfScene();
  bool hasScene() const { return m_scene != nullptr; }

  // Load an HDR environment map. Returns true on success.
  bool loadHdr(const std::filesystem::path& path);
  bool hasHdr() const { return m_hdrIbl.isValid(); }

  // Notify the rasterizer that the output image changed (e.g. after resize).
  void setOutImage(const VkDescriptorImageInfo& imageInfo);

  // Draw GLTF scene + sky/environment.
  // On entry:  colorImage in VK_IMAGE_LAYOUT_GENERAL, depthImage in VK_IMAGE_LAYOUT_GENERAL
  // On return: colorImage in VK_IMAGE_LAYOUT_GENERAL, depthImage in VK_IMAGE_LAYOUT_GENERAL
  void draw(VkCommandBuffer  cmd,
            const VkExtent2D size,
            VkImage          colorImage,
            VkImageView      colorImageView,
            VkImage          depthImage,
            VkImageView      depthImageView,
            const glm::mat4& view,
            const glm::mat4& proj);

  // Public parameters (set before draw())
  shaderio::SkyPhysicalParameters skyParams{};
  shaderio::EnvSystem             envSystem     = shaderio::EnvSystem::eSky;
  float                           envIntensity  = 1.0f;
  float                           envRotation   = 0.0f;
  float                           envBlur       = 0.0f;
  shaderio::DebugMethod           debugMethod   = shaderio::DebugMethod::eNone;
  bool                            useSolidBackground = false;
  glm::vec3                       backgroundColor   = {0.0f, 0.0f, 0.0f};

private:
  void createDescriptorSet();
  void destroyDescriptorSet();
  void createPipelineLayout();
  void destroyPipelineLayout();
  void createShaders();
  void destroyShaders();
  void updateSceneTextures();
  void updateHdrTextures();
  void uploadFrameInfo(VkCommandBuffer cmd, const glm::mat4& view, const glm::mat4& proj);
  void renderNodes(VkCommandBuffer cmd, const std::vector<uint32_t>& nodeIDs);

  nvapp::Application*      m_app{};
  nvvk::ResourceAllocator* m_alloc{};
  nvvk::SamplerPool*       m_samplerPool{};
  VkDevice                 m_device{};
  VkFormat                 m_colorFormat{};
  VkFormat                 m_depthFormat{};

  // Command pool for one-shot commands (HDR loading, scene upload)
  VkCommandPool m_transientCmdPool{};

  // GLTF scene
  std::unique_ptr<nvvkgltf::Scene> m_scene;
  nvvkgltf::SceneVk                m_sceneVk;

  // Environment
  nvvk::HdrIbl         m_hdrIbl;
  nvshaders::HdrEnvDome m_hdrDome;
  nvshaders::SkyPhysical m_skyPhysical;

  // Per-frame GPU buffers (updated every draw)
  nvvk::Buffer m_frameInfoBuffer;
  nvvk::Buffer m_skyParamsBuffer;

  // Descriptor set (set 0): scene textures, cube maps, HDR textures
  static constexpr uint32_t k_maxTextures = 512;
  nvvk::DescriptorBindings  m_descBindings;
  VkDescriptorSetLayout     m_descSetLayout{};
  VkDescriptorPool          m_descPool{};
  VkDescriptorSet           m_descSet{};

  // Pipeline layout and VkShaderEXT objects
  VkPipelineLayout            m_pipelineLayout{};
  nvvk::GraphicsPipelineState m_dynamicPipeline;
  VkShaderEXT                 m_vertexShader{};
  VkShaderEXT                 m_fragmentShader{};
  VkShaderEXT                 m_wireframeShader{};

  // Raster push constant (addresses set per-draw; IDs set per-node)
  shaderio::GltfRasterPushConstant m_pushConst{};
};

}  // namespace vk_gaussian_splatting
