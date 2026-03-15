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

#include "gltf_rasterizer.h"

#include <nvutils/logger.hpp>
#include <nvvk/check_error.hpp>
#include <nvvk/commands.hpp>
#include <nvvk/debug_util.hpp>
#include <nvvk/default_structs.hpp>
#include <nvvk/helpers.hpp>
#include <nvvk/mipmaps.hpp>

// Pre-compiled SPIR-V shaders embedded as C headers
#include "gltf_raster.slang.h"
#include "sky_physical.slang.h"
#include "hdr_prefilter_diffuse.slang.h"
#include "hdr_prefilter_glossy.slang.h"
#include "hdr_integrate_brdf.slang.h"
#include "hdr_dome.slang.h"

namespace vk_gaussian_splatting {

void GltfRasterizer::init(nvapp::Application*      app,
                          nvvk::ResourceAllocator* alloc,
                          nvvk::SamplerPool*       samplerPool,
                          VkFormat                 colorFormat,
                          VkFormat                 depthFormat)
{
  m_app         = app;
  m_alloc       = alloc;
  m_samplerPool = samplerPool;
  m_device      = alloc->getDevice();
  m_colorFormat = colorFormat;
  m_depthFormat = depthFormat;

  m_transientCmdPool = nvvk::createTransientCommandPool(m_device, app->getQueue(0).familyIndex);

  m_sceneVk.init(m_alloc, m_samplerPool);
  m_hdrIbl.init(m_alloc, m_samplerPool);
  m_hdrDome.init(m_alloc, m_samplerPool, m_app->getQueue(0));
  m_skyPhysical.init(m_alloc, std::span(sky_physical_slang));

  // Allocate persistent GPU buffers for frame info and sky params
  NVVK_CHECK(m_alloc->createBuffer(m_frameInfoBuffer, sizeof(shaderio::GltfSceneFrameInfo),
      VK_BUFFER_USAGE_2_UNIFORM_BUFFER_BIT_KHR | VK_BUFFER_USAGE_2_SHADER_DEVICE_ADDRESS_BIT_KHR | VK_BUFFER_USAGE_2_TRANSFER_DST_BIT_KHR));
  NVVK_DBG_NAME(m_frameInfoBuffer.buffer);

  NVVK_CHECK(m_alloc->createBuffer(m_skyParamsBuffer, sizeof(shaderio::SkyPhysicalParameters),
      VK_BUFFER_USAGE_2_UNIFORM_BUFFER_BIT_KHR | VK_BUFFER_USAGE_2_SHADER_DEVICE_ADDRESS_BIT_KHR | VK_BUFFER_USAGE_2_TRANSFER_DST_BIT_KHR));
  NVVK_DBG_NAME(m_skyParamsBuffer.buffer);

  createDescriptorSet();
  createPipelineLayout();
  createShaders();

  // Configure blend equations for the dynamic pipeline
  m_dynamicPipeline.colorBlendEquations[0].alphaBlendOp        = VK_BLEND_OP_ADD;
  m_dynamicPipeline.colorBlendEquations[0].colorBlendOp        = VK_BLEND_OP_ADD;
  m_dynamicPipeline.colorBlendEquations[0].srcAlphaBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
  m_dynamicPipeline.colorBlendEquations[0].dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
  m_dynamicPipeline.colorBlendEquations[0].srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
  m_dynamicPipeline.colorBlendEquations[0].dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
  m_dynamicPipeline.rasterizationState.depthBiasEnable         = VK_TRUE;
  m_dynamicPipeline.rasterizationState.depthBiasConstantFactor = -1.0f;
  m_dynamicPipeline.rasterizationState.depthBiasSlopeFactor    = 1.0f;
}

void GltfRasterizer::deinit()
{
  if(!m_device)
    return;

  vkDeviceWaitIdle(m_device);

  unloadGltfScene();

  destroyShaders();
  destroyPipelineLayout();
  destroyDescriptorSet();

  m_alloc->destroyBuffer(m_frameInfoBuffer);
  m_alloc->destroyBuffer(m_skyParamsBuffer);
  m_frameInfoBuffer = {};
  m_skyParamsBuffer = {};

  m_skyPhysical.deinit();
  m_hdrDome.deinit();
  m_hdrIbl.deinit();
  m_sceneVk.deinit();

  vkDestroyCommandPool(m_device, m_transientCmdPool, nullptr);
  m_transientCmdPool = VK_NULL_HANDLE;

  m_app         = {};
  m_alloc       = {};
  m_samplerPool = {};
  m_device      = {};
}

bool GltfRasterizer::loadGltfScene(const std::filesystem::path& path)
{
  unloadGltfScene();

  auto scene = std::make_unique<nvvkgltf::Scene>();
  if(!scene->load(path))
  {
    LOGW("GltfRasterizer: failed to load %s\n", path.string().c_str());
    return false;
  }

  m_scene = std::move(scene);

  VkCommandBuffer cmd{};
  nvvk::beginSingleTimeCommands(cmd, m_device, m_transientCmdPool);

  nvvk::StagingUploader staging;
  staging.init(m_alloc, true);

  // Upload geometry and textures (no ray tracing)
  m_sceneVk.create(cmd, staging, *m_scene, /*generateMipmaps=*/true, /*enableRayTracing=*/false);

  staging.cmdUploadAppended(cmd);
  nvvk::endSingleTimeCommands(cmd, m_device, m_transientCmdPool, m_app->getQueue(0).queue);
  staging.deinit();

  updateSceneTextures();
  return true;
}

void GltfRasterizer::unloadGltfScene()
{
  if(!m_scene)
    return;

  vkDeviceWaitIdle(m_device);
  m_sceneVk.destroy();
  m_scene.reset();
}

bool GltfRasterizer::loadHdr(const std::filesystem::path& path)
{
  VkCommandBuffer cmd{};
  nvvk::beginSingleTimeCommands(cmd, m_device, m_transientCmdPool);

  nvvk::StagingUploader staging;
  staging.init(m_alloc, true);

  m_hdrIbl.destroyEnvironment();
  m_hdrIbl.loadEnvironment(cmd, staging, path, /*enableMipmaps=*/true);

  staging.cmdUploadAppended(cmd);

  VkExtent2D hdrSize = m_hdrIbl.getHdrImageSize();
  if(hdrSize.width > 1 && hdrSize.height > 1)
  {
    nvvk::cmdGenerateMipmaps(cmd, m_hdrIbl.getHdrImage().image, hdrSize, nvvk::mipLevels(hdrSize));
  }

  nvvk::endSingleTimeCommands(cmd, m_device, m_transientCmdPool, m_app->getQueue(0).queue);
  staging.deinit();

  if(!m_hdrIbl.isValid())
  {
    LOGW("GltfRasterizer: failed to load HDR %s\n", path.string().c_str());
    return false;
  }

  // Pre-filter the HDR into diffuse/glossy cube maps and BRDF LUT
  m_hdrDome.create(m_hdrIbl.getDescriptorSet(), m_hdrIbl.getDescriptorSetLayout(),
                   std::span(hdr_prefilter_diffuse_slang), std::span(hdr_prefilter_glossy_slang),
                   std::span(hdr_integrate_brdf_slang), std::span(hdr_dome_slang));

  updateHdrTextures();
  return true;
}

void GltfRasterizer::setOutImage(const VkDescriptorImageInfo& imageInfo)
{
  m_hdrDome.setOutImage(imageInfo);
}

void GltfRasterizer::draw(VkCommandBuffer  cmd,
                          const VkExtent2D size,
                          VkImage          colorImage,
                          VkImageView      colorImageView,
                          VkImage          depthImage,
                          VkImageView      depthImageView,
                          const glm::mat4& view,
                          const glm::mat4& proj)
{
  NVVK_DBG_SCOPE(cmd);

  uploadFrameInfo(cmd, view, proj);

  // Render sky/environment background while color image is still in GENERAL layout
  // (sky compute and hdr dome both require storage image / GENERAL)
  const VkDescriptorImageInfo colorStorageInfo{
      .sampler     = VK_NULL_HANDLE,
      .imageView   = colorImageView,
      .imageLayout = VK_IMAGE_LAYOUT_GENERAL,
  };
  if(!useSolidBackground)
  {
    if(envSystem == shaderio::EnvSystem::eSky)
    {
      m_skyPhysical.runCompute(cmd, size, view, proj, skyParams, colorStorageInfo);
    }
    else if(envSystem == shaderio::EnvSystem::eHdr && m_hdrIbl.isValid())
    {
      // Update output image for this frame (may differ between double-buffered gBuffers)
      m_hdrDome.setOutImage(colorStorageInfo);
      m_hdrDome.draw(cmd, view, proj, size, glm::vec4(envIntensity), envRotation, envBlur);
    }
  }

  // Transition to attachment layouts for rasterization
  nvvk::cmdImageMemoryBarrier(cmd, {colorImage, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL});
  nvvk::cmdImageMemoryBarrier(cmd, {depthImage, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
                                    {VK_IMAGE_ASPECT_DEPTH_BIT, 0, VK_REMAINING_MIP_LEVELS, 0, VK_REMAINING_ARRAY_LAYERS}});

  // Color load op: LOAD if sky/HDR already rendered, CLEAR if solid background
  VkRenderingAttachmentInfo colorAttachment = DEFAULT_VkRenderingAttachmentInfo;
  colorAttachment.imageView                 = colorImageView;
  colorAttachment.loadOp = useSolidBackground ? VK_ATTACHMENT_LOAD_OP_CLEAR : VK_ATTACHMENT_LOAD_OP_LOAD;
  colorAttachment.clearValue = {{{backgroundColor.x, backgroundColor.y, backgroundColor.z, 0.f}}};

  VkRenderingAttachmentInfo depthAttachment = DEFAULT_VkRenderingAttachmentInfo;
  depthAttachment.imageView                 = depthImageView;
  depthAttachment.clearValue                = {.depthStencil = DEFAULT_VkClearDepthStencilValue};

  VkRenderingInfo renderingInfo      = DEFAULT_VkRenderingInfo;
  renderingInfo.renderArea           = DEFAULT_VkRect2D(size);
  renderingInfo.colorAttachmentCount = 1;
  renderingInfo.pColorAttachments    = &colorAttachment;
  renderingInfo.pDepthAttachment     = &depthAttachment;

  vkCmdBeginRendering(cmd, &renderingInfo);

  if(m_scene)
  {
    // Set push constant BDA pointers (same for all nodes)
    m_pushConst.frameInfo = reinterpret_cast<shaderio::GltfSceneFrameInfo*>(m_frameInfoBuffer.address);
    m_pushConst.skyParams = reinterpret_cast<shaderio::SkyPhysicalParameters*>(m_skyParamsBuffer.address);
    m_pushConst.gltfScene = reinterpret_cast<shaderio::GltfScene*>(m_sceneVk.sceneDesc().address);

    vkCmdPushConstants(cmd, m_pipelineLayout, VK_SHADER_STAGE_ALL_GRAPHICS, 0,
                       sizeof(shaderio::GltfRasterPushConstant), &m_pushConst);

    // Apply all VkShaderEXT dynamic state
    m_dynamicPipeline.cmdApplyAllStates(cmd);
    m_dynamicPipeline.cmdSetViewportAndScissor(cmd, size);
    m_dynamicPipeline.cmdBindShaders(cmd, {.vertex = m_vertexShader, .fragment = m_fragmentShader});
    vkCmdSetDepthTestEnable(cmd, VK_TRUE);

    // Vertex input: position only (vec3 per vertex)
    const auto bindingDesc = std::to_array<VkVertexInputBindingDescription2EXT>({{
        .sType     = VK_STRUCTURE_TYPE_VERTEX_INPUT_BINDING_DESCRIPTION_2_EXT,
        .binding   = 0,
        .stride    = sizeof(glm::vec3),
        .inputRate = VK_VERTEX_INPUT_RATE_VERTEX,
        .divisor   = 1,
    }});
    const auto attrDesc = std::to_array<VkVertexInputAttributeDescription2EXT>({{
        .sType    = VK_STRUCTURE_TYPE_VERTEX_INPUT_ATTRIBUTE_DESCRIPTION_2_EXT,
        .location = 0,
        .binding  = 0,
        .format   = VK_FORMAT_R32G32B32_SFLOAT,
        .offset   = 0,
    }});
    vkCmdSetVertexInputEXT(cmd, uint32_t(bindingDesc.size()), bindingDesc.data(),
                           uint32_t(attrDesc.size()), attrDesc.data());

    // Bind descriptor set 0 (scene textures, cube maps, HDR textures)
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_pipelineLayout, 0, 1, &m_descSet, 0, nullptr);

    // Solid objects: back-face culling + depth bias
    vkCmdSetCullMode(cmd, VK_CULL_MODE_BACK_BIT);
    vkCmdSetDepthBias(cmd, -1.0f, 0.0f, 1.0f);
    renderNodes(cmd, m_scene->getShadedNodes(nvvkgltf::Scene::eRasterSolid));

    // Double-sided: no culling, no depth bias
    vkCmdSetCullMode(cmd, VK_CULL_MODE_NONE);
    vkCmdSetDepthBias(cmd, 0.0f, 0.0f, 0.0f);
    renderNodes(cmd, m_scene->getShadedNodes(nvvkgltf::Scene::eRasterSolidDoubleSided));

    // Alpha-blended objects
    VkBool32 blendEnable = VK_TRUE;
    vkCmdSetColorBlendEnableEXT(cmd, 0, 1, &blendEnable);
    renderNodes(cmd, m_scene->getShadedNodes(nvvkgltf::Scene::eRasterBlend));
  }

  vkCmdEndRendering(cmd);

  // Transition back to GENERAL so the caller (GS pass) can work with the images
  nvvk::cmdImageMemoryBarrier(cmd, {colorImage, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL});
  nvvk::cmdImageMemoryBarrier(cmd, {depthImage, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL,
                                    {VK_IMAGE_ASPECT_DEPTH_BIT, 0, VK_REMAINING_MIP_LEVELS, 0, VK_REMAINING_ARRAY_LAYERS}});
}

// -- Private methods --

void GltfRasterizer::createDescriptorSet()
{
  // Binding 0: all scene textures (bindless array)
  m_descBindings.addBinding(GLTF_BINDING_TEXTURES, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                            k_maxTextures, VK_SHADER_STAGE_ALL, nullptr,
                            VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT
                                | VK_DESCRIPTOR_BINDING_UPDATE_UNUSED_WHILE_PENDING_BIT
                                | VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT);

  // Binding 1: cube maps (2: diffuse + glossy)
  m_descBindings.addBinding(GLTF_BINDING_TEXTURES_CUBE, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                            2, VK_SHADER_STAGE_ALL, nullptr,
                            VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT
                                | VK_DESCRIPTOR_BINDING_UPDATE_UNUSED_WHILE_PENDING_BIT
                                | VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT);

  // Binding 2: HDR 2D textures (2: HDR image + LUT BRDF)
  m_descBindings.addBinding(GLTF_BINDING_TEXTURES_HDR, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                            2, VK_SHADER_STAGE_ALL, nullptr,
                            VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT
                                | VK_DESCRIPTOR_BINDING_UPDATE_UNUSED_WHILE_PENDING_BIT
                                | VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT);

  NVVK_CHECK(m_descBindings.createDescriptorSetLayout(m_device, VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT,
                                                      &m_descSetLayout));
  NVVK_DBG_NAME(m_descSetLayout);

  std::vector<VkDescriptorPoolSize> poolSizes = m_descBindings.calculatePoolSizes();
  VkDescriptorPoolCreateInfo        poolInfo{
      .sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
      .flags         = VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT | VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT,
      .maxSets       = 4,
      .poolSizeCount = uint32_t(poolSizes.size()),
      .pPoolSizes    = poolSizes.data(),
  };
  NVVK_CHECK(vkCreateDescriptorPool(m_device, &poolInfo, nullptr, &m_descPool));
  NVVK_DBG_NAME(m_descPool);

  VkDescriptorSetAllocateInfo allocInfo{
      .sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
      .descriptorPool     = m_descPool,
      .descriptorSetCount = 1,
      .pSetLayouts        = &m_descSetLayout,
  };
  NVVK_CHECK(vkAllocateDescriptorSets(m_device, &allocInfo, &m_descSet));
  NVVK_DBG_NAME(m_descSet);
}

void GltfRasterizer::destroyDescriptorSet()
{
  if(m_descPool)
    vkDestroyDescriptorPool(m_device, m_descPool, nullptr);
  if(m_descSetLayout)
    vkDestroyDescriptorSetLayout(m_device, m_descSetLayout, nullptr);
  m_descPool      = VK_NULL_HANDLE;
  m_descSetLayout = VK_NULL_HANDLE;
  m_descSet       = VK_NULL_HANDLE;
  m_descBindings.clear();
}

void GltfRasterizer::createPipelineLayout()
{
  const VkPushConstantRange pcRange{
      .stageFlags = VK_SHADER_STAGE_ALL_GRAPHICS,
      .offset     = 0,
      .size       = sizeof(shaderio::GltfRasterPushConstant),
  };
  VkPipelineLayoutCreateInfo pipelineLayoutInfo{
      .sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
      .setLayoutCount         = 1,
      .pSetLayouts            = &m_descSetLayout,
      .pushConstantRangeCount = 1,
      .pPushConstantRanges    = &pcRange,
  };
  NVVK_CHECK(vkCreatePipelineLayout(m_device, &pipelineLayoutInfo, nullptr, &m_pipelineLayout));
  NVVK_DBG_NAME(m_pipelineLayout);
}

void GltfRasterizer::destroyPipelineLayout()
{
  if(m_pipelineLayout)
    vkDestroyPipelineLayout(m_device, m_pipelineLayout, nullptr);
  m_pipelineLayout = VK_NULL_HANDLE;
}

void GltfRasterizer::createShaders()
{
  const VkPushConstantRange pcRange{
      .stageFlags = VK_SHADER_STAGE_ALL_GRAPHICS,
      .offset     = 0,
      .size       = sizeof(shaderio::GltfRasterPushConstant),
  };

  VkShaderCreateInfoEXT shaderInfo{
      .sType                  = VK_STRUCTURE_TYPE_SHADER_CREATE_INFO_EXT,
      .stage                  = VK_SHADER_STAGE_VERTEX_BIT,
      .nextStage              = VK_SHADER_STAGE_FRAGMENT_BIT,
      .codeType               = VK_SHADER_CODE_TYPE_SPIRV_EXT,
      .codeSize               = gltf_raster_slang_sizeInBytes,
      .pCode                  = gltf_raster_slang,
      .pName                  = "vertexMain",
      .setLayoutCount         = 1,
      .pSetLayouts            = &m_descSetLayout,
      .pushConstantRangeCount = 1,
      .pPushConstantRanges    = &pcRange,
  };

  NVVK_CHECK(vkCreateShadersEXT(m_device, 1, &shaderInfo, nullptr, &m_vertexShader));
  NVVK_DBG_NAME(m_vertexShader);

  shaderInfo.stage     = VK_SHADER_STAGE_FRAGMENT_BIT;
  shaderInfo.nextStage = 0;
  shaderInfo.pName     = "fragmentMain";
  NVVK_CHECK(vkCreateShadersEXT(m_device, 1, &shaderInfo, nullptr, &m_fragmentShader));
  NVVK_DBG_NAME(m_fragmentShader);

  shaderInfo.pName = "fragmentWireframeMain";
  NVVK_CHECK(vkCreateShadersEXT(m_device, 1, &shaderInfo, nullptr, &m_wireframeShader));
  NVVK_DBG_NAME(m_wireframeShader);
}

void GltfRasterizer::destroyShaders()
{
  vkDestroyShaderEXT(m_device, m_vertexShader, nullptr);
  vkDestroyShaderEXT(m_device, m_fragmentShader, nullptr);
  vkDestroyShaderEXT(m_device, m_wireframeShader, nullptr);
  m_vertexShader    = VK_NULL_HANDLE;
  m_fragmentShader  = VK_NULL_HANDLE;
  m_wireframeShader = VK_NULL_HANDLE;
}

void GltfRasterizer::updateSceneTextures()
{
  if(!m_scene)
    return;

  uint32_t count = m_sceneVk.nbTextures();
  if(count == 0)
    return;
  if(count > k_maxTextures)
  {
    LOGW("GltfRasterizer: scene has %u textures, exceeds limit of %u\n", count, k_maxTextures);
    count = k_maxTextures;
  }

  VkWriteDescriptorSet writeSet = m_descBindings.getWriteSet(GLTF_BINDING_TEXTURES, m_descSet, 0, count);
  nvvk::WriteSetContainer write{};
  write.append(writeSet, m_sceneVk.textures().data());
  vkUpdateDescriptorSets(m_device, write.size(), write.data(), 0, nullptr);
}

void GltfRasterizer::updateHdrTextures()
{
  if(!m_hdrIbl.isValid())
    return;

  const std::vector<nvvk::Image>& hdrTextures = m_hdrDome.getTextures();
  nvvk::WriteSetContainer         write{};

  // HDR 2D: index 0 = raw HDR image, index 1 = LUT BRDF
  VkWriteDescriptorSet hdrWrite =
      m_descBindings.getWriteSet(GLTF_BINDING_TEXTURES_HDR, m_descSet, HDR_IMAGE_INDEX, 1);
  write.append(hdrWrite, m_hdrIbl.getHdrImage());

  hdrWrite.dstArrayElement = HDR_LUT_INDEX;
  write.append(hdrWrite, hdrTextures[2]);  // lutBrdf is index 2 in getTextures()

  // Cube maps: index 0 = diffuse, index 1 = glossy
  VkWriteDescriptorSet cubeWrite =
      m_descBindings.getWriteSet(GLTF_BINDING_TEXTURES_CUBE, m_descSet, 0, 2);
  write.append(cubeWrite, hdrTextures.data());  // diffuse=0, glossy=1

  vkUpdateDescriptorSets(m_device, write.size(), write.data(), 0, nullptr);
}

void GltfRasterizer::uploadFrameInfo(VkCommandBuffer cmd, const glm::mat4& view, const glm::mat4& proj)
{
  shaderio::GltfSceneFrameInfo frameInfo{};
  frameInfo.viewMatrix     = view;
  frameInfo.projInv        = glm::inverse(proj);
  frameInfo.viewInv        = glm::inverse(view);
  frameInfo.viewProjMatrix = proj * view;
  frameInfo.isOrthographic = 0;
  frameInfo.envRotation    = envRotation;
  frameInfo.envBlur        = envBlur;
  frameInfo.envIntensity   = envIntensity;
  frameInfo.useSolidBackground = useSolidBackground ? 1 : 0;
  frameInfo.backgroundColor    = backgroundColor;
  frameInfo.environmentType    = static_cast<int>(envSystem);
  frameInfo.debugMethod        = debugMethod;

  vkCmdUpdateBuffer(cmd, m_frameInfoBuffer.buffer, 0, sizeof(frameInfo), &frameInfo);
  vkCmdUpdateBuffer(cmd, m_skyParamsBuffer.buffer, 0, sizeof(skyParams), &skyParams);

  VkMemoryBarrier barrier{
      .sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
      .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
      .dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_UNIFORM_READ_BIT,
  };
  vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                       VK_PIPELINE_STAGE_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT
                           | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                       0, 1, &barrier, 0, nullptr, 0, nullptr);
}

void GltfRasterizer::renderNodes(VkCommandBuffer cmd, const std::vector<uint32_t>& nodeIDs)
{
  NVVK_DBG_SCOPE(cmd);

  const VkDeviceSize offsets{0};

  const std::vector<nvvkgltf::RenderNode>&      renderNodes = m_scene->getRenderNodes();
  const std::vector<nvvkgltf::RenderPrimitive>& subMeshes   = m_scene->getRenderPrimitives();

  struct NodeSpecificConstants
  {
    int32_t materialID;
    int32_t renderNodeID;
    int32_t renderPrimID;
  };

  const uint32_t offset = static_cast<uint32_t>(offsetof(shaderio::GltfRasterPushConstant, materialID));

  for(const uint32_t nodeID : nodeIDs)
  {
    const nvvkgltf::RenderNode& renderNode = renderNodes[nodeID];
    if(!renderNode.visible)
      continue;

    const nvvkgltf::RenderPrimitive& subMesh = subMeshes[renderNode.renderPrimID];

    NodeSpecificConstants nodeConst{
        .materialID   = renderNode.materialID,
        .renderNodeID = static_cast<int32_t>(nodeID),
        .renderPrimID = renderNode.renderPrimID,
    };
    vkCmdPushConstants(cmd, m_pipelineLayout, VK_SHADER_STAGE_ALL_GRAPHICS, offset,
                       sizeof(NodeSpecificConstants), &nodeConst);

    vkCmdBindVertexBuffers(cmd, 0, 1, &m_sceneVk.vertexBuffers()[renderNode.renderPrimID].position.buffer, &offsets);
    vkCmdBindIndexBuffer(cmd, m_sceneVk.indices()[renderNode.renderPrimID].buffer, 0, VK_INDEX_TYPE_UINT32);
    vkCmdDrawIndexed(cmd, subMesh.indexCount, 1, 0, 0, 0);
  }
}

}  // namespace vk_gaussian_splatting
