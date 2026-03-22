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

#ifndef _GAUSSIAN_SPLAT_RENDERER_H_
#define _GAUSSIAN_SPLAT_RENDERER_H_

#include <vulkan/vulkan_core.h>
#include <glm/vec3.hpp>
#include <glm/mat4x4.hpp>
#include <glm/vec2.hpp>
#include <vk_radix_sort.h>

#include <nvvk/resource_allocator.hpp>
#include <nvvk/staging.hpp>
#include <nvvk/sampler_pool.hpp>
#include <nvvk/physical_device.hpp>
#include <nvslang/slang.hpp>
#include <nvapp/application.hpp>

#include "shaderio.h"
#include "parameters.h"
#include "splat_set.h"
#include "splat_set_vk.h"
#include "splat_sorter_async.h"
#include "light_set_vk.h"
#include "camera_set.h"

namespace vk_gaussian_splatting
{

// Self-contained Gaussian splat renderer.
// Owns all GPU resources for sorting and drawing splats.
// Follows the same lifecycle pattern as GltfRasterizer: init / draw / deinit.
class GaussianSplatRenderer
{
public:
	GaussianSplatRenderer()  = default;
	~GaussianSplatRenderer() = default;

	// Initialize Vulkan resources. Call once at startup.
	void init(nvapp::Application*      app,
	          nvvk::ResourceAllocator* alloc,
	          nvvk::StagingUploader*   uploader,
	          VkSampler*               sampler,
	          VkFormat                 colorFormat,
	          VkFormat                 depthFormat);

	// Release all Vulkan resources.
	void deinit();

	// --- Scene data lifecycle ---

	// Upload splat data into VRAM and build shaders/pipelines.
	// Returns false if shader compilation fails.
	bool loadSplatData(SplatSet& splatSet);

	// Unload all scene-related GPU resources.
	void unloadSplatData();

	// Returns true when splat data is loaded and shaders are valid.
	bool isReady() const { return m_shaders.valid; }

	// Returns the splat count from the loaded data, or 0.
	uint32_t splatCount() const { return m_splatCount; }

	// --- Deferred update requests (set from UI, executed before draw) ---

	void requestUpdateShaders()      { m_requestUpdateShaders = true; }
	void requestUpdateSplatData()    { m_requestUpdateSplatData = true; }
	void requestUpdateLightsBuffer() { m_requestUpdateLightsBuffer = true; }

	// Set camera model for shader compilation (CAMERA_PINHOLE or CAMERA_FISHEYE).
	void setCameraModel(int32_t model) { m_cameraModel = model; }

	// Process pending update requests. May call vkDeviceWaitIdle.
	void processUpdateRequests(SplatSet& splatSet);

	// --- Per-frame drawing ---

	// Parameters needed for a draw call, supplied by the caller each frame.
	struct DrawParams
	{
		glm::mat4  view;
		glm::mat4  proj;
		glm::vec3  eye;
		glm::vec3  center;
		glm::vec3  up;
		float      fovRad;
		glm::vec2  nearFar;
		VkExtent2D viewportSize;
		int32_t    cameraModel;     // CAMERA_PINHOLE or CAMERA_FISHEYE
		float      focusDist;
		float      aperture;
		bool       hasGltfScene;    // enables depth test for splats
	};

	// Collect readback values from the previous frame's indirect params.
	// Call at the start of the frame before processUpdateRequests.
	void collectReadback();

	// Upload the frame UBO and perform GPU/CPU sorting.
	// Call after processUpdateRequests, before the render pass.
	void prepareDraw(VkCommandBuffer cmd, const DrawParams& params);

	// Issue draw commands for splat primitives.
	// Must be called inside an active render pass with viewport/scissor set.
	void drawSplatPrimitives(VkCommandBuffer cmd, const DrawParams& params);

	// Readback indirect parameters for statistics. Call after the render pass.
	void postDraw(VkCommandBuffer cmd);

	// --- Accessors for UI and app element ---

	SplatSetVk&       getSplatSetVk()       { return m_splatSetVk; }
	const SplatSetVk& getSplatSetVk() const { return m_splatSetVk; }

	LightSetVk&       getLightSet()       { return m_lightSet; }
	const LightSetVk& getLightSet() const { return m_lightSet; }

	const shaderio::IndirectParams& getIndirectReadback() const { return m_indirectReadback; }

private:
	void initRendererBuffers(uint32_t splatCount);
	void deinitRendererBuffers();

	void updateSlangMacros();
	bool compileSlangShader(const std::string& filename, VkShaderModule& module);
	bool initShaders();
	void deinitShaders();

	void initPipelines();
	void deinitPipelines();

	void updateAndUploadFrameInfoUBO(VkCommandBuffer cmd, const DrawParams& params);
	void processSortingOnGPU(VkCommandBuffer cmd, uint32_t splatCount);
	void tryConsumeAndUploadCpuSortingResult(VkCommandBuffer cmd, uint32_t splatCount);

	void collectReadBackValuesIfNeeded();
	void readBackIndirectParametersIfNeeded(VkCommandBuffer cmd);

	// --- Stored init context ---
	nvapp::Application*      m_app{nullptr};
	nvvk::ResourceAllocator* m_alloc{nullptr};
	nvvk::StagingUploader*   m_uploader{nullptr};
	VkDevice                 m_device{VK_NULL_HANDLE};
	VkSampler*               m_sampler{nullptr};
	VkFormat                 m_colorFormat{VK_FORMAT_UNDEFINED};
	VkFormat                 m_depthFormat{VK_FORMAT_UNDEFINED};
	nvvk::PhysicalDeviceInfo m_physicalDeviceInfo;

	// --- Scene data (VRAM) ---
	SplatSetVk m_splatSetVk{};
	LightSetVk m_lightSet{};
	uint32_t   m_splatCount{0};

	// --- External data references (set by loadSplatData) ---
	std::vector<float>* m_positions{nullptr};  // CPU positions for async sort
	int32_t             m_cameraModel{0};      // camera model for shader macros

	// --- Push constant ---
	shaderio::PushConstant m_pcRaster{};

	// --- Update request flags ---
	bool m_requestUpdateSplatData{false};
	bool m_requestUpdateShaders{false};
	bool m_requestUpdateLightsBuffer{false};

	// --- Indirect draw / readback ---
	nvvk::Buffer                m_indirect;
	nvvk::Buffer                m_indirectReadbackHost;
	shaderio::IndirectParams    m_indirectReadback{};
	bool                        m_canCollectReadback{false};

	// --- Quad geometry ---
	nvvk::Buffer m_quadVertices;
	nvvk::Buffer m_quadIndices;

	// --- Sorting ---
	SplatSorterAsync         m_cpuSorter;
	std::vector<uint32_t>    m_splatIndices;
	VrdxSorter               m_gpuSorter{VK_NULL_HANDLE};
	nvvk::Buffer             m_splatIndicesHost;
	nvvk::Buffer             m_splatIndicesDevice;
	nvvk::Buffer             m_splatDistancesDevice;
	nvvk::Buffer             m_vrdxStorageDevice;

	// --- Camera state (written by prepareDraw for CPU sort) ---
	glm::vec3 m_eye{};
	glm::vec3 m_center{};
	glm::vec3 m_up{};

	// --- Shaders ---
	std::vector<std::pair<std::string, std::string>> m_shaderMacros;
	nvslang::SlangCompiler m_slangCompiler{};

	struct Shaders
	{
		VkShaderModule distShader{};
		VkShaderModule meshShader{};
		VkShaderModule vertexShader{};
		VkShaderModule fragmentShader{};
		VkShaderModule threedgutMeshShader{};
		VkShaderModule threedgutFragmentShader{};
		std::vector<VkShaderModule*> modules{};
		bool valid{false};
	} m_shaders;

	// --- Pipelines ---
	VkPipeline            m_computePipelineGsDistCull{VK_NULL_HANDLE};
	VkPipeline            m_graphicsPipelineGsVert{VK_NULL_HANDLE};
	VkPipeline            m_graphicsPipelineGsMesh{VK_NULL_HANDLE};
	VkPipeline            m_graphicsPipeline3dgutMesh{VK_NULL_HANDLE};
	VkPipelineLayout      m_pipelineLayout{VK_NULL_HANDLE};
	VkDescriptorSetLayout m_descriptorSetLayout{VK_NULL_HANDLE};
	VkDescriptorSet       m_descriptorSet{VK_NULL_HANDLE};
	VkDescriptorPool      m_descriptorPool{VK_NULL_HANDLE};

	// --- UBO ---
	nvvk::Buffer m_frameInfoBuffer;
};

} // namespace vk_gaussian_splatting

#endif
