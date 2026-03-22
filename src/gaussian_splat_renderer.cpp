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

#include "gaussian_splat_renderer.h"
#include "utilities.h"

#include <nvvk/check_error.hpp>
#include <nvvk/descriptors.hpp>
#include <nvvk/graphics_pipeline.hpp>

#include <glm/gtc/quaternion.hpp>

#include <chrono>
#include <iostream>

namespace vk_gaussian_splatting
{

void GaussianSplatRenderer::init(nvapp::Application*      app,
                                  nvvk::ResourceAllocator* alloc,
                                  nvvk::StagingUploader*   uploader,
                                  VkSampler*               sampler,
                                  VkFormat                 colorFormat,
                                  VkFormat                 depthFormat)
{
	m_app         = app;
	m_alloc       = alloc;
	m_uploader    = uploader;
	m_device      = app->getDevice();
	m_sampler     = sampler;
	m_colorFormat = colorFormat;
	m_depthFormat = depthFormat;

	m_physicalDeviceInfo.init(app->getPhysicalDevice(), VK_API_VERSION_1_4);

	// Setting up the Slang compiler
	m_slangCompiler.addSearchPaths(getShaderDirs());
	m_slangCompiler.defaultTarget();
	m_slangCompiler.defaultOptions();
	m_slangCompiler.addOption({
		slang::CompilerOptionName::MatrixLayoutRow, {slang::CompilerOptionValueKind::Int, 1}
	});
	m_slangCompiler.addOption({
		slang::CompilerOptionName::DebugInformation,
		{slang::CompilerOptionValueKind::Int, SLANG_DEBUG_INFO_LEVEL_MAXIMAL}
	});
	m_slangCompiler.addOption({
		slang::CompilerOptionName::Optimization,
		{slang::CompilerOptionValueKind::Int, SLANG_OPTIMIZATION_LEVEL_DEFAULT}
	});

	// Init GPU data holder
	m_splatSetVk.init(app, alloc, uploader, m_sampler, &m_physicalDeviceInfo);

	// Start async CPU sort thread
	m_cpuSorter.initialize();
}

void GaussianSplatRenderer::deinit()
{
	m_cpuSorter.shutdown();
	unloadSplatData();
	m_splatSetVk.deinit();
}

bool GaussianSplatRenderer::loadSplatData(SplatSet& splatSet)
{
	vkDeviceWaitIdle(m_device);

	m_splatCount = (uint32_t)splatSet.size();
	m_splatIndices.resize(m_splatCount);
	m_positions = &splatSet.positions;

	m_lightSet.init(m_app, m_alloc, m_uploader);

	if(!initShaders())
		return false;

	initRendererBuffers(m_splatCount);
	m_splatSetVk.initDataStorage(splatSet, prmData.dataStorage, prmData.shFormat);
	initPipelines();

	return true;
}

void GaussianSplatRenderer::unloadSplatData()
{
	vkDeviceWaitIdle(m_device);

	m_canCollectReadback = false;
	m_splatCount = 0;
	m_splatSetVk.resetTransform();
	m_splatSetVk.deinitDataStorage();
	m_lightSet.deinit();
	deinitShaders();
	deinitPipelines();
	deinitRendererBuffers();
}

void GaussianSplatRenderer::processUpdateRequests(SplatSet& splatSet)
{
	bool needUpdate = m_requestUpdateSplatData || m_requestUpdateShaders || m_requestUpdateLightsBuffer;

	if(!splatSet.size() || !needUpdate)
		return;

	vkDeviceWaitIdle(m_device);

	if(m_requestUpdateSplatData || m_requestUpdateShaders)
	{
		deinitPipelines();
		deinitShaders();

		if(m_requestUpdateSplatData)
		{
			m_splatSetVk.deinitDataStorage();
			m_splatSetVk.initDataStorage(splatSet, prmData.dataStorage, prmData.shFormat);
		}

		if(initShaders())
		{
			initPipelines();
		}
	}

	if(m_requestUpdateLightsBuffer)
	{
		m_lightSet.updateBuffer();
		m_requestUpdateLightsBuffer = false;
	}

	m_requestUpdateSplatData = m_requestUpdateShaders = m_requestUpdateLightsBuffer = false;
}

void GaussianSplatRenderer::collectReadback()
{
	collectReadBackValuesIfNeeded();
}

void GaussianSplatRenderer::prepareDraw(VkCommandBuffer cmd, const DrawParams& params)
{
	if(!m_shaders.valid || !m_splatCount)
		return;

	// Store camera state for CPU sort
	m_eye    = params.eye;
	m_center = params.center;
	m_up     = params.up;

	updateAndUploadFrameInfoUBO(cmd, params);

	if(prmRaster.sortingMethod == SORTING_GPU_SYNC_RADIX)
	{
		processSortingOnGPU(cmd, m_splatCount);
	}
	else
	{
		tryConsumeAndUploadCpuSortingResult(cmd, m_splatCount);
	}
}

void GaussianSplatRenderer::drawSplatPrimitives(VkCommandBuffer cmd, const DrawParams& params)
{
	NVVK_DBG_SCOPE(cmd);

	bool needDepth = ((prmRaster.sortingMethod != SORTING_GPU_SYNC_RADIX) && prmRender.opacityGaussianDisabled)
		|| params.hasGltfScene;

	// Model transform
	m_pcRaster.modelMatrix = m_splatSetVk.transform;
	m_pcRaster.modelMatrixInverse = m_splatSetVk.transformInverse;
	glm::mat3 rotScale = glm::mat3(m_splatSetVk.transform);
	m_pcRaster.modelMatrixRotScaleInverse = glm::inverse(rotScale);

	vkCmdPushConstants(cmd, m_pipelineLayout,
	                   VK_SHADER_STAGE_COMPUTE_BIT | VK_SHADER_STAGE_MESH_BIT_EXT | VK_SHADER_STAGE_VERTEX_BIT |
	                   VK_SHADER_STAGE_FRAGMENT_BIT,
	                   0, sizeof(shaderio::PushConstant), &m_pcRaster);

	if(prmSelectedPipeline == PIPELINE_VERT)
	{
		vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_graphicsPipelineGsVert);
		vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_pipelineLayout, 0, 1, &m_descriptorSet, 0,
		                        nullptr);

		vkCmdSetDepthWriteEnable(cmd, (VkBool32)needDepth);
		vkCmdSetDepthTestEnable(cmd, (VkBool32)needDepth);

		const VkDeviceSize offsets{0};
		vkCmdBindIndexBuffer(cmd, m_quadIndices.buffer, 0, VK_INDEX_TYPE_UINT16);
		vkCmdBindVertexBuffers(cmd, 0, 1, &m_quadVertices.buffer, &offsets);
		if(prmRaster.sortingMethod != SORTING_GPU_SYNC_RADIX)
		{
			vkCmdBindVertexBuffers(cmd, 1, 1, &m_splatIndicesDevice.buffer, &offsets);
			vkCmdDrawIndexed(cmd, 6, m_splatCount, 0, 0, 0);
		}
		else
		{
			vkCmdBindVertexBuffers(cmd, 1, 1, &m_splatIndicesDevice.buffer, &offsets);
			vkCmdDrawIndexedIndirect(cmd, m_indirect.buffer, 0, 1, sizeof(VkDrawIndexedIndirectCommand));
		}
	}
	else
	{
		if(prmSelectedPipeline == PIPELINE_MESH)
			vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_graphicsPipelineGsMesh);
		if(prmSelectedPipeline == PIPELINE_MESH_3DGUT)
			vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_graphicsPipeline3dgutMesh);

		vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_pipelineLayout, 0, 1, &m_descriptorSet, 0,
		                        nullptr);

		vkCmdSetDepthWriteEnable(cmd, (VkBool32)needDepth);
		vkCmdSetDepthTestEnable(cmd, (VkBool32)needDepth);

		if(prmRaster.sortingMethod != SORTING_GPU_SYNC_RADIX)
		{
			vkCmdDrawMeshTasksEXT(
				cmd, (prmFrame.splatCount + prmRaster.meshShaderWorkgroupSize - 1) / prmRaster.meshShaderWorkgroupSize,
				1, 1);
		}
		else
		{
			vkCmdDrawMeshTasksIndirectEXT(cmd, m_indirect.buffer, offsetof(shaderio::IndirectParams, groupCountX),
			                              1, sizeof(VkDrawMeshTasksIndirectCommandEXT));
		}
	}
}

void GaussianSplatRenderer::postDraw(VkCommandBuffer cmd)
{
	readBackIndirectParametersIfNeeded(cmd);
}

// ---------------------------------------------------------------------------
// Private methods
// ---------------------------------------------------------------------------

void GaussianSplatRenderer::updateAndUploadFrameInfoUBO(VkCommandBuffer cmd, const DrawParams& params)
{
	NVVK_DBG_SCOPE(cmd);

	prmFrame.splatCount = m_splatCount;
	prmFrame.lightCount = int32_t(m_lightSet.size());

	prmFrame.cameraPosition = params.eye;
	prmFrame.viewMatrix     = params.view;
	prmFrame.viewInverse    = glm::inverse(params.view);

	prmFrame.fovRad            = params.fovRad;
	prmFrame.nearFar           = params.nearFar;
	prmFrame.projectionMatrix  = params.proj;
	prmFrame.projInverse       = glm::inverse(params.proj);

	const float devicePixelRatio = 1.0f;
	prmFrame.orthoZoom               = 1.0f;
	prmFrame.orthographicMode        = 0;
	prmFrame.viewport                = glm::vec2(params.viewportSize.width * devicePixelRatio,
	                                              params.viewportSize.height * devicePixelRatio);
	prmFrame.basisViewport           = glm::vec2(1.0f / params.viewportSize.width, 1.0f / params.viewportSize.height);
	prmFrame.inverseFocalAdjustment  = 1.0f;

	if(params.cameraModel == CAMERA_FISHEYE && prmSelectedPipeline != PIPELINE_VERT && prmSelectedPipeline != PIPELINE_MESH)
	{
		prmFrame.focal = glm::vec2(1.0, -1.0) * prmFrame.viewport / prmFrame.fovRad;
	}
	else
	{
		const float focalLengthX = prmFrame.projectionMatrix[0][0] * 0.5f * devicePixelRatio * params.viewportSize.width;
		const float focalLengthY = prmFrame.projectionMatrix[1][1] * 0.5f * devicePixelRatio * params.viewportSize.height;
		prmFrame.focal = glm::vec2(focalLengthX, focalLengthY);
	}

	// Camera pose, used by unscented transform
	prmFrame.viewTrans = prmFrame.viewMatrix[3];
	glm::quat viewQuat = glm::quat_cast(prmFrame.viewMatrix);
	prmFrame.viewQuat  = glm::vec4(viewQuat.x, viewQuat.y, viewQuat.z, viewQuat.w);

	prmFrame.focusDist = params.focusDist;
	prmFrame.aperture  = params.aperture;

	vkCmdUpdateBuffer(cmd, m_frameInfoBuffer.buffer, 0, sizeof(shaderio::FrameInfo), &prmFrame);

	VkMemoryBarrier barrier = {VK_STRUCTURE_TYPE_MEMORY_BARRIER};
	barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
	barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_UNIFORM_READ_BIT;

	vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_2_TRANSFER_BIT,
	                     VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT
	                     | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_MESH_SHADER_BIT_EXT,
	                     0, 1, &barrier, 0, NULL, 0, NULL);
}

void GaussianSplatRenderer::tryConsumeAndUploadCpuSortingResult(VkCommandBuffer cmd, const uint32_t splatCount)
{
	NVVK_DBG_SCOPE(cmd);

	bool newIndexAvailable = false;

	if(!prmRender.opacityGaussianDisabled)
	{
		auto status = m_cpuSorter.getStatus();
		if(status != SplatSorterAsync::E_SORTING)
		{
			if(status == SplatSorterAsync::E_SORTED)
			{
				m_cpuSorter.consume(m_splatIndices);
				newIndexAvailable = true;
			}

			m_cpuSorter.sortAsync(glm::normalize(m_center - m_eye), m_eye, *m_positions,
			                      m_splatSetVk.transform, prmRaster.cpuLazySort);
		}
	}
	else
	{
		bool refill = (m_splatIndices.size() != splatCount);
		if(refill)
		{
			m_splatIndices.resize(splatCount);
			for(uint32_t i = 0; i < splatCount; ++i)
				m_splatIndices[i] = i;
			newIndexAvailable = true;
		}
	}

	if(newIndexAvailable)
	{
		memcpy(m_splatIndicesHost.mapping, m_splatIndices.data(), m_splatIndices.size() * sizeof(uint32_t));
		VkBufferCopy bc{.srcOffset = 0, .dstOffset = 0, .size = splatCount * sizeof(uint32_t)};
		vkCmdCopyBuffer(cmd, m_splatIndicesHost.buffer, m_splatIndicesDevice.buffer, 1, &bc);

		VkMemoryBarrier barrier = {VK_STRUCTURE_TYPE_MEMORY_BARRIER};
		barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
		barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

		vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_2_TRANSFER_BIT,
		                     VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT |
		                     VK_PIPELINE_STAGE_2_MESH_SHADER_BIT_EXT,
		                     0, 1, &barrier, 0, NULL, 0, NULL);
	}
}

void GaussianSplatRenderer::processSortingOnGPU(VkCommandBuffer cmd, const uint32_t splatCount)
{
	NVVK_DBG_SCOPE(cmd);

	// 1. reset the draw indirect parameters and counters
	{
		const shaderio::IndirectParams drawIndexedIndirectParams;
		vkCmdUpdateBuffer(cmd, m_indirect.buffer, 0, sizeof(shaderio::IndirectParams),
		                  (void*)&drawIndexedIndirectParams);

		VkMemoryBarrier barrier = {VK_STRUCTURE_TYPE_MEMORY_BARRIER};
		barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
		barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_INDIRECT_COMMAND_READ_BIT;

		vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_2_TRANSFER_BIT,
		                     VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_MESH_SHADER_BIT_EXT |
		                     VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT,
		                     0, 1, &barrier, 0, NULL, 0, NULL);
	}

	VkMemoryBarrier barrier = {VK_STRUCTURE_TYPE_MEMORY_BARRIER};
	barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
	barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_INDIRECT_COMMAND_READ_BIT;

	// 2. invoke the distance compute shader
	{
		vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_computePipelineGsDistCull);
		vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelineLayout, 0, 1, &m_descriptorSet, 0,
		                        nullptr);

		m_pcRaster.modelMatrix = m_splatSetVk.transform;
		m_pcRaster.modelMatrixInverse = m_splatSetVk.transformInverse;

		vkCmdPushConstants(cmd, m_pipelineLayout,
		                   VK_SHADER_STAGE_COMPUTE_BIT | VK_SHADER_STAGE_MESH_BIT_EXT | VK_SHADER_STAGE_VERTEX_BIT |
		                   VK_SHADER_STAGE_FRAGMENT_BIT,
		                   0, sizeof(shaderio::PushConstant), &m_pcRaster);

		vkCmdDispatch(cmd, (splatCount + prmRaster.distShaderWorkgroupSize - 1) / prmRaster.distShaderWorkgroupSize,
		              1, 1);

		vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
		                     VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_MESH_SHADER_BIT_EXT |
		                     VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT,
		                     0, 1, &barrier, 0, NULL, 0, NULL);
	}

	// 3. invoke the radix sort
	{
		vrdxCmdSortKeyValueIndirect(cmd, m_gpuSorter, splatCount, m_indirect.buffer,
		                            offsetof(shaderio::IndirectParams, instanceCount),
		                            m_splatDistancesDevice.buffer, 0,
		                            m_splatIndicesDevice.buffer, 0, m_vrdxStorageDevice.buffer, 0, 0, 0);

		vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_TRANSFER_BIT,
		                     VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_2_MESH_SHADER_BIT_EXT |
		                     VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT,
		                     0, 1, &barrier, 0, NULL, 0, NULL);
	}
}

void GaussianSplatRenderer::collectReadBackValuesIfNeeded()
{
	if(m_indirectReadbackHost.buffer != VK_NULL_HANDLE && prmRaster.sortingMethod == SORTING_GPU_SYNC_RADIX &&
		m_canCollectReadback)
	{
		std::memcpy((void*)&m_indirectReadback, (void*)m_indirectReadbackHost.mapping,
		            sizeof(shaderio::IndirectParams));
	}
}

void GaussianSplatRenderer::readBackIndirectParametersIfNeeded(VkCommandBuffer cmd)
{
	NVVK_DBG_SCOPE(cmd);

	if(m_indirectReadbackHost.buffer != VK_NULL_HANDLE && prmRaster.sortingMethod == SORTING_GPU_SYNC_RADIX)
	{
		VkMemoryBarrier barrier = {VK_STRUCTURE_TYPE_MEMORY_BARRIER};
		barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
		barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;

		vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_TRANSFER_BIT, 0, 1,
		                     &barrier, 0, NULL, 0, NULL);

		VkBufferCopy bc{.srcOffset = 0, .dstOffset = 0, .size = sizeof(shaderio::IndirectParams)};
		vkCmdCopyBuffer(cmd, m_indirect.buffer, m_indirectReadbackHost.buffer, 1, &bc);

		m_canCollectReadback = true;
	}
}

void GaussianSplatRenderer::updateSlangMacros()
{
	m_shaderMacros =
	{
		{"PIPELINE", std::to_string(prmSelectedPipeline)},
		{"CAMERA_TYPE", std::to_string(m_cameraModel)},
		{"VISUALIZE", std::to_string((int)prmRender.visualize)},
		{"DISABLE_OPACITY_GAUSSIAN", std::to_string((int)prmRender.opacityGaussianDisabled)},
		{"FRUSTUM_CULLING_MODE", std::to_string(prmRaster.frustumCulling)},
		{"ORTHOGRAPHIC_MODE", "0"},
		{"SHOW_SH_ONLY", std::to_string((int)prmRender.showShOnly)},
		{"MAX_SH_DEGREE", std::to_string(prmRender.maxShDegree)},
		{"DATA_STORAGE", std::to_string(prmData.dataStorage)},
		{"SH_FORMAT", std::to_string(prmData.shFormat)},
		{"POINT_CLOUD_MODE", std::to_string((int)prmRaster.pointCloudModeEnabled)},
		{"USE_BARYCENTRIC", std::to_string((int)prmRaster.fragmentBarycentric)},
		{"WIREFRAME", std::to_string((int)prmRender.wireframe)},
		{"DISTANCE_COMPUTE_WORKGROUP_SIZE", std::to_string((int)prmRaster.distShaderWorkgroupSize)},
		{"RASTER_MESH_WORKGROUP_SIZE", std::to_string((int)prmRaster.meshShaderWorkgroupSize)},
		{"MS_ANTIALIASING", std::to_string((int)prmRaster.msAntialiasing)},
		{"EXTENT_METHOD", std::to_string((int)prmRaster.extentProjection)},
		{"KERNEL_DEGREE", std::to_string(KERNEL_DEGREE_QUADRATIC)},
		{"KERNEL_MIN_RESPONSE", std::to_string(0.0113f)},
	};

	m_slangCompiler.clearMacros();
	for(auto& macro : m_shaderMacros)
		m_slangCompiler.addMacro({macro.first.c_str(), macro.second.c_str()});
}

bool GaussianSplatRenderer::compileSlangShader(const std::string& filename, VkShaderModule& module)
{
	if(!m_slangCompiler.compileFile(filename))
		return false;

	if(module != VK_NULL_HANDLE)
		vkDestroyShaderModule(m_device, module, nullptr);

	VkShaderModuleCreateInfo createInfo{
		.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
		.codeSize = m_slangCompiler.getSpirvSize(),
		.pCode = m_slangCompiler.getSpirv()
	};

	if(m_slangCompiler.getSpirvSize() == 0)
	{
		std::cerr << "\033[31m" << "Missing entry point in shader " << std::endl;
		std::cerr << filename << "\033[0m" << std::endl;
		return false;
	}
	NVVK_CHECK(vkCreateShaderModule(m_device, &createInfo, nullptr, &module));
	NVVK_DBG_NAME(module);

	m_shaders.modules.emplace_back(&module);
	return true;
}

bool GaussianSplatRenderer::initShaders()
{
	auto startTime = std::chrono::high_resolution_clock::now();

	bool success = true;
	updateSlangMacros();

	success &= compileSlangShader("dist.comp.slang", m_shaders.distShader);
	success &= compileSlangShader("threedgs_raster.vert.slang", m_shaders.vertexShader);
	success &= compileSlangShader("threedgs_raster.mesh.slang", m_shaders.meshShader);
	success &= compileSlangShader("threedgs_raster.frag.slang", m_shaders.fragmentShader);
	success &= compileSlangShader("threedgut_raster.mesh.slang", m_shaders.threedgutMeshShader);
	success &= compileSlangShader("threedgut_raster.frag.slang", m_shaders.threedgutFragmentShader);

	if(!success)
		return (m_shaders.valid = false);

	auto endTime = std::chrono::high_resolution_clock::now();
	long long buildTime = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();
	std::cout << "Shaders updated in " << buildTime << "ms" << std::endl;

	return (m_shaders.valid = true);
}

void GaussianSplatRenderer::deinitShaders()
{
	for(auto& shader : m_shaders.modules)
	{
		vkDestroyShaderModule(m_device, *shader, nullptr);
		*shader = VK_NULL_HANDLE;
	}

	m_shaders.valid = false;
	m_shaders.modules.clear();
}

void GaussianSplatRenderer::initPipelines()
{
	nvvk::DescriptorBindings bindings;

	bindings.addBinding(BINDING_FRAME_INFO_UBO, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_ALL);
	bindings.addBinding(BINDING_DISTANCES_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_ALL);
	bindings.addBinding(BINDING_INDICES_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_ALL);
	bindings.addBinding(BINDING_INDIRECT_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_ALL);

	if(prmData.dataStorage == STORAGE_TEXTURES)
	{
		bindings.addBinding(BINDING_CENTERS_TEXTURE, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_ALL);
		bindings.addBinding(BINDING_SCALES_TEXTURE, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_ALL);
		bindings.addBinding(BINDING_ROTATIONS_TEXTURE, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_ALL);
		bindings.addBinding(BINDING_COVARIANCES_TEXTURE, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_ALL);
		bindings.addBinding(BINDING_COLORS_TEXTURE, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_ALL);
		bindings.addBinding(BINDING_SH_TEXTURE, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_ALL);
	}
	else
	{
		bindings.addBinding(BINDING_CENTERS_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_ALL);
		bindings.addBinding(BINDING_SCALES_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_ALL);
		bindings.addBinding(BINDING_ROTATIONS_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_ALL);
		bindings.addBinding(BINDING_COVARIANCES_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_ALL);
		bindings.addBinding(BINDING_COLORS_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_ALL);
		bindings.addBinding(BINDING_SH_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_ALL);
	}

	bindings.addBinding(BINDING_LIGHT_SET, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_ALL);

	const VkPushConstantRange pcRanges = {
		VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT
		| VK_SHADER_STAGE_MESH_BIT_EXT | VK_SHADER_STAGE_COMPUTE_BIT,
		0, sizeof(shaderio::PushConstant)
	};

	NVVK_CHECK(bindings.createDescriptorSetLayout(m_device, 0, &m_descriptorSetLayout));
	NVVK_DBG_NAME(m_descriptorSetLayout);

	std::vector<VkDescriptorPoolSize> poolSize;
	bindings.appendPoolSizes(poolSize);
	VkDescriptorPoolCreateInfo poolInfo = {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
		.maxSets = 1,
		.poolSizeCount = uint32_t(poolSize.size()),
		.pPoolSizes = poolSize.data(),
	};
	NVVK_CHECK(vkCreateDescriptorPool(m_device, &poolInfo, nullptr, &m_descriptorPool));
	NVVK_DBG_NAME(m_descriptorPool);

	VkDescriptorSetAllocateInfo allocInfo = {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
		.descriptorPool = m_descriptorPool,
		.descriptorSetCount = 1,
		.pSetLayouts = &m_descriptorSetLayout,
	};
	NVVK_CHECK(vkAllocateDescriptorSets(m_device, &allocInfo, &m_descriptorSet));
	NVVK_DBG_NAME(m_descriptorSet);

	VkPipelineLayoutCreateInfo plCreateInfo{
		.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
		.setLayoutCount = 1,
		.pSetLayouts = &m_descriptorSetLayout,
		.pushConstantRangeCount = 1,
		.pPushConstantRanges = &pcRanges,
	};
	NVVK_CHECK(vkCreatePipelineLayout(m_device, &plCreateInfo, nullptr, &m_pipelineLayout));
	NVVK_DBG_NAME(m_pipelineLayout);

	// Write descriptors
	nvvk::WriteSetContainer writeContainer;

	writeContainer.append(bindings.getWriteSet(BINDING_FRAME_INFO_UBO, m_descriptorSet), m_frameInfoBuffer);
	writeContainer.append(bindings.getWriteSet(BINDING_DISTANCES_BUFFER, m_descriptorSet), m_splatDistancesDevice);
	writeContainer.append(bindings.getWriteSet(BINDING_INDICES_BUFFER, m_descriptorSet), m_splatIndicesDevice);
	writeContainer.append(bindings.getWriteSet(BINDING_INDIRECT_BUFFER, m_descriptorSet), m_indirect);

	if(prmData.dataStorage == STORAGE_TEXTURES)
	{
		writeContainer.append(bindings.getWriteSet(BINDING_CENTERS_TEXTURE, m_descriptorSet), m_splatSetVk.centersMap);
		writeContainer.append(bindings.getWriteSet(BINDING_SCALES_TEXTURE, m_descriptorSet), m_splatSetVk.scalesMap);
		writeContainer.append(bindings.getWriteSet(BINDING_ROTATIONS_TEXTURE, m_descriptorSet), m_splatSetVk.rotationsMap);
		writeContainer.append(bindings.getWriteSet(BINDING_COVARIANCES_TEXTURE, m_descriptorSet), m_splatSetVk.covariancesMap);
		writeContainer.append(bindings.getWriteSet(BINDING_COLORS_TEXTURE, m_descriptorSet), m_splatSetVk.colorsMap);
		writeContainer.append(bindings.getWriteSet(BINDING_SH_TEXTURE, m_descriptorSet), m_splatSetVk.sphericalHarmonicsMap);
	}
	else
	{
		writeContainer.append(bindings.getWriteSet(BINDING_CENTERS_BUFFER, m_descriptorSet), m_splatSetVk.centersBuffer);
		writeContainer.append(bindings.getWriteSet(BINDING_SCALES_BUFFER, m_descriptorSet), m_splatSetVk.scalesBuffer);
		writeContainer.append(bindings.getWriteSet(BINDING_ROTATIONS_BUFFER, m_descriptorSet), m_splatSetVk.rotationsBuffer);
		writeContainer.append(bindings.getWriteSet(BINDING_COVARIANCES_BUFFER, m_descriptorSet), m_splatSetVk.covariancesBuffer);
		writeContainer.append(bindings.getWriteSet(BINDING_COLORS_BUFFER, m_descriptorSet), m_splatSetVk.colorsBuffer);
		if(m_splatSetVk.sphericalHarmonicsBuffer.buffer != NULL)
			writeContainer.append(bindings.getWriteSet(BINDING_SH_BUFFER, m_descriptorSet), m_splatSetVk.sphericalHarmonicsBuffer);
	}

	if(m_lightSet.size())
		writeContainer.append(bindings.getWriteSet(BINDING_LIGHT_SET, m_descriptorSet), m_lightSet.lightsBuffer);

	vkUpdateDescriptorSets(m_device, static_cast<uint32_t>(writeContainer.size()), writeContainer.data(), 0, nullptr);

	// Compute pipeline for distance & culling
	{
		VkComputePipelineCreateInfo pipelineInfo{
			.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
			.stage = {
				.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
				.stage = VK_SHADER_STAGE_COMPUTE_BIT,
				.module = m_shaders.distShader,
				.pName = "main",
			},
			.layout = m_pipelineLayout,
		};
		vkCreateComputePipelines(m_device, {}, 1, &pipelineInfo, nullptr, &m_computePipelineGsDistCull);
		NVVK_DBG_NAME(m_computePipelineGsDistCull);
	}

	// Graphics pipelines
	{
		nvvk::GraphicsPipelineState pipelineState;
		pipelineState.rasterizationState.cullMode = VK_CULL_MODE_NONE;

		pipelineState.colorBlendEnables[0] = VK_TRUE;
		pipelineState.colorBlendEquations[0].alphaBlendOp = VK_BLEND_OP_ADD;
		pipelineState.colorBlendEquations[0].colorBlendOp = VK_BLEND_OP_ADD;
		pipelineState.colorBlendEquations[0].srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
		pipelineState.colorBlendEquations[0].dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
		pipelineState.colorBlendEquations[0].srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
		pipelineState.colorBlendEquations[0].dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;

		pipelineState.depthStencilState.depthWriteEnable = VK_FALSE;
		pipelineState.depthStencilState.depthTestEnable = VK_FALSE;

		// Mesh shader pipeline for 3DGS
		{
			nvvk::GraphicsPipelineCreator creator;
			creator.pipelineInfo.layout = m_pipelineLayout;
			creator.colorFormats = {m_colorFormat};
			creator.renderingState.depthAttachmentFormat = m_depthFormat;
			creator.dynamicStateValues.push_back(VK_DYNAMIC_STATE_DEPTH_WRITE_ENABLE);
			creator.dynamicStateValues.push_back(VK_DYNAMIC_STATE_DEPTH_TEST_ENABLE);
			creator.addShader(VK_SHADER_STAGE_MESH_BIT_EXT, "main", m_shaders.meshShader);
			creator.addShader(VK_SHADER_STAGE_FRAGMENT_BIT, "main_mesh", m_shaders.fragmentShader);
			creator.createGraphicsPipeline(m_device, nullptr, pipelineState, &m_graphicsPipelineGsMesh);
			NVVK_DBG_NAME(m_graphicsPipelineGsMesh);
		}

		// Mesh shader pipeline for 3DGUT
		{
			nvvk::GraphicsPipelineCreator creator;
			creator.pipelineInfo.layout = m_pipelineLayout;
			creator.colorFormats = {m_colorFormat};
			creator.renderingState.depthAttachmentFormat = m_depthFormat;
			creator.dynamicStateValues.push_back(VK_DYNAMIC_STATE_DEPTH_WRITE_ENABLE);
			creator.dynamicStateValues.push_back(VK_DYNAMIC_STATE_DEPTH_TEST_ENABLE);
			creator.addShader(VK_SHADER_STAGE_MESH_BIT_EXT, "main", m_shaders.threedgutMeshShader);
			creator.addShader(VK_SHADER_STAGE_FRAGMENT_BIT, "main", m_shaders.threedgutFragmentShader);
			creator.createGraphicsPipeline(m_device, nullptr, pipelineState, &m_graphicsPipeline3dgutMesh);
			NVVK_DBG_NAME(m_graphicsPipeline3dgutMesh);
		}

		// Vertex shader pipeline for 3DGS
		{
			const auto BINDING_ATTR_POSITION = 0;
			const auto BINDING_ATTR_SPLAT_INDEX = 1;

			pipelineState.vertexBindings = {
				{.binding = BINDING_ATTR_POSITION, .stride = 3 * sizeof(float), .divisor = 1},
				{.binding = BINDING_ATTR_SPLAT_INDEX, .stride = sizeof(uint32_t), .inputRate = VK_VERTEX_INPUT_RATE_INSTANCE, .divisor = 1}
			};
			pipelineState.vertexAttributes = {
				{.location = ATTRIBUTE_LOC_POSITION, .binding = BINDING_ATTR_POSITION, .format = VK_FORMAT_R32G32B32_SFLOAT, .offset = 0},
				{.location = ATTRIBUTE_LOC_SPLAT_INDEX, .binding = BINDING_ATTR_SPLAT_INDEX, .format = VK_FORMAT_R32_UINT, .offset = 0}
			};

			nvvk::GraphicsPipelineCreator creator;
			creator.pipelineInfo.layout = m_pipelineLayout;
			creator.colorFormats = {m_colorFormat};
			creator.renderingState.depthAttachmentFormat = m_depthFormat;
			creator.dynamicStateValues.push_back(VK_DYNAMIC_STATE_DEPTH_WRITE_ENABLE);
			creator.dynamicStateValues.push_back(VK_DYNAMIC_STATE_DEPTH_TEST_ENABLE);
			creator.addShader(VK_SHADER_STAGE_VERTEX_BIT, "main", m_shaders.vertexShader);
			creator.addShader(VK_SHADER_STAGE_FRAGMENT_BIT, "main", m_shaders.fragmentShader);
			creator.createGraphicsPipeline(m_device, nullptr, pipelineState, &m_graphicsPipelineGsVert);
			NVVK_DBG_NAME(m_graphicsPipelineGsVert);
		}
	}
}

void GaussianSplatRenderer::deinitPipelines()
{
	if(m_graphicsPipelineGsVert == VK_NULL_HANDLE)
		return;

	TEST_DESTROY_AND_RESET(m_graphicsPipelineGsVert, vkDestroyPipeline(m_device, m_graphicsPipelineGsVert, nullptr));
	TEST_DESTROY_AND_RESET(m_graphicsPipelineGsMesh, vkDestroyPipeline(m_device, m_graphicsPipelineGsMesh, nullptr));
	TEST_DESTROY_AND_RESET(m_graphicsPipeline3dgutMesh, vkDestroyPipeline(m_device, m_graphicsPipeline3dgutMesh, nullptr));
	TEST_DESTROY_AND_RESET(m_computePipelineGsDistCull, vkDestroyPipeline(m_device, m_computePipelineGsDistCull, nullptr));
	TEST_DESTROY_AND_RESET(m_pipelineLayout, vkDestroyPipelineLayout(m_device, m_pipelineLayout, nullptr));
	TEST_DESTROY_AND_RESET(m_descriptorSetLayout, vkDestroyDescriptorSetLayout(m_device, m_descriptorSetLayout, nullptr));
	TEST_DESTROY_AND_RESET(m_descriptorPool, vkDestroyDescriptorPool(m_device, m_descriptorPool, nullptr));
}

void GaussianSplatRenderer::initRendererBuffers(uint32_t splatCount)
{
	// Sorting buffers
	{
		VrdxSorterCreateInfo gpuSorterInfo{
			.physicalDevice = m_app->getPhysicalDevice(), .device = m_app->getDevice()
		};
		vrdxCreateSorter(&gpuSorterInfo, &m_gpuSorter);

		const VkDeviceSize bufferSize = ((splatCount * sizeof(uint32_t) + 15) / 16) * 16;

		m_alloc->createBuffer(m_splatIndicesHost, bufferSize, VK_BUFFER_USAGE_2_TRANSFER_SRC_BIT,
		                      VMA_MEMORY_USAGE_AUTO_PREFER_HOST,
		                      VMA_ALLOCATION_CREATE_MAPPED_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT);

		m_alloc->createBuffer(m_splatIndicesDevice, bufferSize,
		                      VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_2_TRANSFER_SRC_BIT |
		                      VK_BUFFER_USAGE_2_TRANSFER_DST_BIT | VK_BUFFER_USAGE_2_SHADER_DEVICE_ADDRESS_BIT |
		                      VK_BUFFER_USAGE_2_VERTEX_BUFFER_BIT,
		                      VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);

		m_alloc->createBuffer(m_splatDistancesDevice, bufferSize,
		                      VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_2_TRANSFER_SRC_BIT |
		                      VK_BUFFER_USAGE_2_TRANSFER_DST_BIT | VK_BUFFER_USAGE_2_SHADER_DEVICE_ADDRESS_BIT |
		                      VK_BUFFER_USAGE_2_VERTEX_BUFFER_BIT,
		                      VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);

		VrdxSorterStorageRequirements requirements;
		vrdxGetSorterKeyValueStorageRequirements(m_gpuSorter, splatCount, &requirements);
		m_alloc->createBuffer(m_vrdxStorageDevice, requirements.size, requirements.usage,
		                      VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);

		NVVK_DBG_NAME(m_splatIndicesHost.buffer);
		NVVK_DBG_NAME(m_splatIndicesDevice.buffer);
		NVVK_DBG_NAME(m_splatDistancesDevice.buffer);
		NVVK_DBG_NAME(m_vrdxStorageDevice.buffer);
	}

	// Indirect parameter buffers
	m_alloc->createBuffer(m_indirect, sizeof(shaderio::IndirectParams),
	                      VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_2_TRANSFER_SRC_BIT
	                      | VK_BUFFER_USAGE_2_TRANSFER_DST_BIT | VK_BUFFER_USAGE_2_INDIRECT_BUFFER_BIT,
	                      VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);

	m_alloc->createBuffer(m_indirectReadbackHost, sizeof(shaderio::IndirectParams),
	                      VK_BUFFER_USAGE_2_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_2_TRANSFER_DST_BIT,
	                      VMA_MEMORY_USAGE_AUTO_PREFER_HOST,
	                      VMA_ALLOCATION_CREATE_MAPPED_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT);

	NVVK_DBG_NAME(m_indirect.buffer);
	NVVK_DBG_NAME(m_indirectReadbackHost.buffer);

	// Quad geometry
	VkCommandBuffer cmd = m_app->createTempCmdBuffer();

	const std::vector<uint16_t> indices = {0, 2, 1, 2, 0, 3};
	const std::vector<float> vertices = {-1.0, -1.0, 0.0, 1.0, -1.0, 0.0, 1.0, 1.0, 0.0, -1.0, 1.0, 0.0};

	m_alloc->createBuffer(m_quadVertices, vertices.size() * sizeof(float), VK_BUFFER_USAGE_2_VERTEX_BUFFER_BIT,
	                      VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);
	m_alloc->createBuffer(m_quadIndices, indices.size() * sizeof(uint16_t), VK_BUFFER_USAGE_2_INDEX_BUFFER_BIT,
	                      VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);

	NVVK_DBG_NAME(m_quadVertices.buffer);
	NVVK_DBG_NAME(m_quadIndices.buffer);

	vkCmdUpdateBuffer(cmd, m_quadVertices.buffer, 0, vertices.size() * sizeof(float), vertices.data());
	vkCmdUpdateBuffer(cmd, m_quadIndices.buffer, 0, indices.size() * sizeof(uint16_t), indices.data());
	m_app->submitAndWaitTempCmdBuffer(cmd);

	// Frame info UBO
	m_alloc->createBuffer(m_frameInfoBuffer, sizeof(shaderio::FrameInfo),
	                      VK_BUFFER_USAGE_2_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_2_TRANSFER_SRC_BIT |
	                      VK_BUFFER_USAGE_2_TRANSFER_DST_BIT,
	                      VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);
	NVVK_DBG_NAME(m_frameInfoBuffer.buffer);
}

void GaussianSplatRenderer::deinitRendererBuffers()
{
	if(m_gpuSorter != VK_NULL_HANDLE)
	{
		vrdxDestroySorter(m_gpuSorter);
		m_gpuSorter = VK_NULL_HANDLE;
	}

	m_alloc->destroyBuffer(m_splatDistancesDevice);
	m_alloc->destroyBuffer(m_splatIndicesDevice);
	m_alloc->destroyBuffer(m_splatIndicesHost);
	m_alloc->destroyBuffer(m_vrdxStorageDevice);

	m_alloc->destroyBuffer(m_indirect);
	m_alloc->destroyBuffer(m_indirectReadbackHost);

	m_alloc->destroyBuffer(m_quadVertices);
	m_alloc->destroyBuffer(m_quadIndices);

	m_alloc->destroyBuffer(m_frameInfoBuffer);
}

} // namespace vk_gaussian_splatting
