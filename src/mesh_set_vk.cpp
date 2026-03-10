/*
 * Copyright (c) 2021-2025, NVIDIA CORPORATION.  All rights reserved.
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
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "mesh_set_vk.h"
#include "utilities.h"

//#define STB_IMAGE_IMPLEMENTATION
//#include <stb/stb_image.h>

#include <nvvk/debug_util.hpp>
#include <nvvk/resource_allocator.hpp>
#include <nvvk/resources.hpp>
#include <nvvk/check_error.hpp>
#include <nvvk/debug_util.hpp>

#include <nvutils/logger.hpp>
#include <nvutils/timers.hpp>

namespace vk_gaussian_splatting {

bool MeshSetVk::loadModel(const std::filesystem::path& filename)
{
  LOGI("Loading File:  %s \n", filename.string().c_str());
  ObjLoader loader;
  if(!loader.load(filename))
    return false;

  // Converting from Srgb to linear
  for(auto& m : loader.m_materials)
  {
    m.ambient  = glm::pow(m.ambient, glm::vec3(2.2f));
    m.diffuse  = glm::pow(m.diffuse, glm::vec3(2.2f));
    m.specular = glm::pow(m.specular, glm::vec3(2.2f));
  }

  Mesh model;
  model.path       = loader.filename.string();
  model.name       = loader.filename.filename().string();
  model.nbIndices  = static_cast<uint32_t>(loader.m_indices.size());
  model.nbVertices = static_cast<uint32_t>(loader.m_vertices.size());
  model.materials  = loader.m_materials;
  model.matNames   = loader.m_matNames;

  // Create the buffers on Device and copy vertices, indices and materials

  VkBufferUsageFlags flag            = VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
  VkBufferUsageFlags rayTracingFlags =  // used also for building acceleration structures
      flag | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;

  NVVK_CHECK(m_alloc->createBuffer(model.vertexBuffer, loader.m_vertices.size() * sizeof(ObjVertex),
                                   VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | rayTracingFlags));
  NVVK_DBG_NAME(model.vertexBuffer.buffer);

  NVVK_CHECK(m_alloc->createBuffer(model.indexBuffer, loader.m_indices.size() * sizeof(uint32_t),
                                   VK_BUFFER_USAGE_INDEX_BUFFER_BIT | rayTracingFlags));
  NVVK_DBG_NAME(model.indexBuffer.buffer);

  NVVK_CHECK(m_alloc->createBuffer(model.materialsBuffer, loader.m_materials.size() * sizeof(ObjMaterial),
                                   VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | rayTracingFlags));
  NVVK_DBG_NAME(model.materialsBuffer.buffer);

  NVVK_CHECK(m_alloc->createBuffer(model.matIndexBuffer, loader.m_matIndices.size() * sizeof(uint32_t),
                                   VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | rayTracingFlags));
  NVVK_DBG_NAME(model.matIndexBuffer.buffer);

  // Intializes the buffers and upload

  VkCommandBuffer cmdBuf = m_app->createTempCmdBuffer();

  NVVK_CHECK(m_uploader->appendBuffer(model.vertexBuffer, 0, std::span(loader.m_vertices)));
  NVVK_CHECK(m_uploader->appendBuffer(model.indexBuffer, 0, std::span(loader.m_indices)));
  NVVK_CHECK(m_uploader->appendBuffer(model.materialsBuffer, 0, std::span(loader.m_materials)));
  NVVK_CHECK(m_uploader->appendBuffer(model.matIndexBuffer, 0, std::span(loader.m_matIndices)));

  m_uploader->cmdUploadAppended(cmdBuf);
  m_app->submitAndWaitTempCmdBuffer(cmdBuf);
  m_uploader->releaseStaging();

  // Keeping transformation matrix of the instance
  Instance instance;
  instance.transform = glm::mat4(1);
  instance.objIndex  = static_cast<uint32_t>(meshes.size());
  computeTransform(instance.scale, instance.rotation, instance.translation, instance.transform, instance.transformInverse);
  instances.push_back(instance);

  // Creating information for device access
  shaderio::ObjDesc desc;
  //desc.txtOffset            = txtOffset;
  desc.vertexAddress        = (shaderio::ObjVertex*)model.vertexBuffer.address;
  desc.indexAddress         = (uint32_t*)model.indexBuffer.address;
  desc.materialAddress      = (shaderio::ObjMaterial*)model.materialsBuffer.address;
  desc.materialIndexAddress = (uint32_t*)model.matIndexBuffer.address;

  // Keeping the obj host model and device description
  meshes.emplace_back(model);
  objectDescriptions.emplace_back(desc);

  return true;
}

void MeshSetVk::updateObjDescriptionBuffer()
{

  if(objectDescriptionsBuffer.buffer != VK_NULL_HANDLE)
    m_alloc->destroyBuffer(objectDescriptionsBuffer);

  if(objectDescriptions.empty())
    return;

  // Create buffer

  NVVK_CHECK(m_alloc->createBuffer(objectDescriptionsBuffer, objectDescriptions.size() * sizeof(shaderio::ObjDesc),
                                   VK_BUFFER_USAGE_STORAGE_BUFFER_BIT));
  NVVK_DBG_NAME(objectDescriptionsBuffer.buffer);

  // Init and upload buffer

  VkCommandBuffer cmdBuf = m_app->createTempCmdBuffer();

  NVVK_CHECK(m_uploader->appendBuffer(objectDescriptionsBuffer, 0, std::span(objectDescriptions)));

  m_uploader->cmdUploadAppended(cmdBuf);
  m_app->submitAndWaitTempCmdBuffer(cmdBuf);
  m_uploader->releaseStaging();
}

void MeshSetVk::updateObjMaterialsBuffer(int modelIndex)
{

  const auto& model = meshes[modelIndex];

  VkCommandBuffer cmdBuf = m_app->createTempCmdBuffer();

  NVVK_CHECK(m_uploader->appendBuffer(model.materialsBuffer, 0, std::span(model.materials)));

  m_uploader->cmdUploadAppended(cmdBuf);
  m_app->submitAndWaitTempCmdBuffer(cmdBuf);
  m_uploader->releaseStaging();
}

void MeshSetVk::deleteInstance(uint32_t instanceId)
{
  // several instances may refer to the same mesh
  // we check if no more instance refer to the mesh after we remove this one
  bool       lastInstanceUsingMesh = true;
  const auto meshIndex             = instances[instanceId].objIndex;

  for(uint32_t i = 0; i < instances.size(); ++i)
  {
    const auto& instance = instances[i];
    if(instanceId != i && instance.objIndex == meshIndex)
    {
      lastInstanceUsingMesh = false;
      break;
    }
  }

  // remove the entry
  instances.erase(instances.begin() + instanceId);
  objectDescriptions.erase(objectDescriptions.begin() + instanceId);

  // free the mesh if needed
  if(lastInstanceUsingMesh)
  {
    // free the buffers
    deinitMeshBuffers(meshes[meshIndex]);
    // remove the entry
    meshes.erase(meshes.begin() + meshIndex);
    // shift the mesh indices in the remaining instances
    for(auto& instance : instances)
    {
      if(instance.objIndex >= meshIndex)
      {
        --instance.objIndex;
      }
    }
  }
}

}  // namespace vk_gaussian_splatting