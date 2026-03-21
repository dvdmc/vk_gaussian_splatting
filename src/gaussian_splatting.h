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
 * SPDX-FileCopyrightText: Copyright (c) 2023-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef _GAUSSIAN_SPLATTING_H_
#define _GAUSSIAN_SPLATTING_H_

#include <iostream>
#include <string>
#include <array>
#include <chrono>
#include <filesystem>
#include <span>
// Important: include Igmlui before Vulkan
// Or undef "Status" before including imgui
#include <imgui/imgui.h>
//
#include <vulkan/vulkan_core.h>
// mathematics
#include <glm/vec3.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtx/string_cast.hpp>
#include <glm/gtx/transform.hpp>
// threading
#include <thread>
#include <condition_variable>
#include <mutex>
// GPU radix sort
#include <vk_radix_sort.h>
//
#include <nvutils/logger.hpp>
#include <nvutils/file_operations.hpp>
#include <nvutils/alignment.hpp>

#include <nvvk/context.hpp>
#include <nvvk/debug_util.hpp>
#include <nvvk/pipeline.hpp>
#include <nvvk/physical_device.hpp>
#include <nvvk/helpers.hpp>
#include <nvvk/gbuffers.hpp>
#include <nvvk/resources.hpp>
#include <nvvk/resource_allocator.hpp>
#include <nvvk/staging.hpp>
#include <nvvk/validation_settings.hpp>
#include <nvvk/sampler_pool.hpp>
#include <nvvk/default_structs.hpp>
#include <nvvk/descriptors.hpp>

#include <nvvkglsl/glsl.hpp>
#include <nvslang/slang.hpp>

#include <nvapp/application.hpp>
#include <nvapp/elem_camera.hpp>
#include <nvapp/elem_sequencer.hpp>
#include <nvapp/elem_default_title.hpp>
#include <nvapp/elem_default_menu.hpp>
//
#include <nvgui/axis.hpp>
#include <nvgui/enum_registry.hpp>
#include <nvgui/property_editor.hpp>
#include <nvgui/file_dialog.hpp>
//
#include <nvgpu_monitor/elem_gpu_monitor.hpp>

// Shared between host and device
#include "shaderio.h"

#include "parameters.h"
#include "utilities.h"
#include "splat_set.h"
#include "splat_set_vk.h"
#include "ply_loader_async.h"
#include "splat_sorter_async.h"
#include "light_set_vk.h"
#include "camera_set.h"
#include "gltf_rasterizer.h"

namespace vk_gaussian_splatting
{
	class GaussianSplatting
	{
	public:
		// Camera manipulator for the interactive display camera.
		// Public so that it can be accessed by main / ElementCamera.
		std::shared_ptr<nvutils::CameraManipulator> cameraManip{};

		struct ReadbackBuffer
		{
			VkBuffer       buffer{VK_NULL_HANDLE};
			VkDeviceMemory memory{VK_NULL_HANDLE};
			VkDeviceSize   size{0};
		};

		// Load a GLTF scene for compositing with splats.
		// Safe to call after application.addElement() (which invokes onAttach).
		void loadGltfScene(const std::filesystem::path& path);

		// Load an HDR environment map.
		// Safe to call after application.addElement() (which invokes onAttach).
		void loadHdr(const std::filesystem::path& path);

		// --- Multi-camera interface ---
		// All methods below are safe to call from the application thread between frames.
		// Do NOT call addCamera / removeCamera / setVirtualCameraResolution from a
		// background thread while onRender is executing.

		// Add a virtual camera with the given pose. Returns its index in the CameraSet
		// (always >= 1; index 0 is the interactive display camera).
		// Allocates a per-camera CPU readback buffer at the current virtual resolution.
		uint64_t addCamera(const Camera& cam);

		// Remove the virtual camera at index (index 0 cannot be removed).
		// Frees the associated readback buffer. Returns false if index is invalid.
		bool removeCamera(uint64_t index);

		// Update the pose of virtual camera at index. Returns false if index is invalid.
		bool setCamera(uint64_t index, const Camera& cam);

		// Read the current pose of camera at index (0 = display camera).
		Camera getCamera(uint64_t index) const;

		// Total number of cameras including the display camera (index 0).
		uint64_t getCameraCount() const;

		// Copy the latest rendered image of virtual camera at index into out (RGBA8, row-major).
		// Thread-safe: acquires a per-camera mutex. Returns false if no image is ready yet
		// or if index is 0 / out of range.
		bool getImage(uint64_t index, std::vector<unsigned char>& out);

		// Resolution used for all virtual camera off-screen renders.
		// Changing this recreates the offscreen GBuffer and all readback buffers.
		// Default: 640x480.
		void       setVirtualCameraResolution(VkExtent2D size);
		VkExtent2D getVirtualCameraResolution() const { return m_virtualCameraResolution; }

	protected:
		GaussianSplatting();

		~GaussianSplatting();

		void onAttach(nvapp::Application* app);

		void onDetach();

		void onResize(VkCommandBuffer cmd, const VkExtent2D& size);

		void onRender(VkCommandBuffer cmd);

		// reset the rendering settings that can
		// be modified by the user interface
		inline void resetRenderSettings()
		{
			resetFrameParameters();
			resetRenderParameters();
			resetRasterParameters();
		}

		// Initializes all that is related to the scene based
		// on current parameters. VRAM Data, shaders, pipelines.
		// Invoked on scene load success.
		bool initAll();

		// Denitializes all that is related to the scene.
		// VRAM Data, shaders, pipelines.
		// Invoked on scene close or on exit.
		void deinitAll();

		// free scene (splat set) from RAM
		void deinitScene();

	private:
		// Per-virtual-camera capture state.
		// unique_ptr because std::mutex is not movable.
		struct CameraCapture
		{
			ReadbackBuffer             buffer;        // GPU→CPU readback buffer
			std::vector<unsigned char> image;         // CPU pixels, valid when ready == true
			bool                       ready = false; // set true at the start of the frame after capture
			std::mutex                 mutex;         // protects image and ready for ROS thread access
		};

		// One entry per virtual camera (capture[i] corresponds to cameraSet preset[i+1]).
		std::vector<std::unique_ptr<CameraCapture>> m_cameraCaptures;

		// Off-screen GBuffer reused for all virtual camera renders each frame.
		nvvk::GBuffer m_offscreenGBuffer;

		// Resolution of all virtual camera renders (independent of the display window).
		VkExtent2D m_virtualCameraResolution{640, 480};

		// Upload prmFrame UBO for a virtual camera (view/proj computed from Camera struct).
		void uploadFrameInfoForCamera(VkCommandBuffer cmd, uint32_t splatCount,
		                              const Camera& cam, VkExtent2D viewport);

		// At the start of each frame, copy the previous frame's readback buffer
		// contents into CameraCapture::image under the per-camera mutex.
		void collectVirtualCameraCaptures();

		// Render all virtual cameras into m_offscreenGBuffer and copy to readback buffers.
		void renderVirtualCameras(VkCommandBuffer cmd, uint32_t splatCount);

		// init the raster pipelines
		void initPipelines();

		// deinit raster pipelines
		void deinitPipelines();

		void initRendererBuffers();

		void deinitRendererBuffers();

		void updateSlangMacros(void);

		bool compileSlangShader(const std::string& filename, VkShaderModule& module);

		bool initShaders(void);

		void deinitShaders(void);

		static VkResult updateReadbackBuffer(VkDevice device, VkPhysicalDevice physicalDevice, VkDeviceSize size, ReadbackBuffer& out);

		/////////////
		// Rendering submethods

		// process eventual update requests comming from UI or benchmark
		// that requires to be performed before a new rendering after a DeviceWaitIdle
		void processUpdateRequests(void);

		// Updates frame information uniform buffer and frame camera info
		void updateAndUploadFrameInfoUBO(VkCommandBuffer cmd, const uint32_t splatCount);

		void tryConsumeAndUploadCpuSortingResult(VkCommandBuffer cmd, const uint32_t splatCount);

		void processSortingOnGPU(VkCommandBuffer cmd, const uint32_t splatCount);

		void drawSplatPrimitives(VkCommandBuffer cmd, const uint32_t splatCount);


		// for statistics display in the UI
		// copy form m_indirectReadbackHost updated at previous frame to m_indirectReadback
		void collectReadBackValuesIfNeeded(void);
		// for statistics display in the UI
		// read back updated indirect parameters from m_indirect into m_indirectReadbackHost
		void readBackIndirectParametersIfNeeded(VkCommandBuffer cmd);


	protected:
		// name of the loaded scene if load is successfull
		std::filesystem::path m_loadedSceneFilename;

		// Paths of loaded GLTF scene and HDR map (set by loadGltfScene / loadHdr)
		std::filesystem::path m_loadedGltfFilename;
		std::filesystem::path m_loadedHdrFilename;

		// scene loader
		PlyLoaderAsync m_plyLoader;
		// 3DGS/3DGRT model in RAM
		SplatSet m_splatSet = {};
		// 3DGS/3DGRT model in VRAM
		SplatSetVk m_splatSetVk = {};
		// GLTF scene rasterizer
		GltfRasterizer m_gltfRasterizer;
		// Set of lights in RAM and VRAM
		LightSetVk m_lightSet = {};
		// Set of cameras in RAM
		CameraSet m_cameraSet = {};

		// Index of the last camera loaded
		uint64_t m_lastLoadedCamera = 0;

		// Push constant for rasterizer
		shaderio::PushConstant m_pcRaster{};

		// counting benchmark steps
		int m_benchmarkId = 0;

		// trigger a rebuild of the data in VRAM (textures or buffers) at next frame
		// also triggers shaders and pipeline rebuild
		bool m_requestUpdateSplatData = false;
		// trigger a rebuild of the shaders and pipelines at next frame
		bool m_requestUpdateShaders = false;
		// trigger the update of light buffer at next frame
		bool m_requestUpdateLightsBuffer = false;

		nvapp::Application* m_app{nullptr};
		nvvk::StagingUploader m_uploader{}; // utility to upload buffers to device
		nvvk::SamplerPool m_samplerPool{}; // The sampler pool, used to create texture samplers
		VkSampler m_sampler{}; // texture sampler (nearest)
		nvvk::ResourceAllocator m_alloc;
		nvvk::PhysicalDeviceInfo m_physicalDeviceInfo;

		glm::vec2 m_viewSize = {0, 0};
		VkFormat m_colorFormat = VK_FORMAT_R8G8B8A8_UNORM; // Color format of the image
		VkFormat m_depthFormat = VK_FORMAT_UNDEFINED; // Depth format of the depth buffer
		VkClearColorValue m_clearColor = {{0.0F, 0.0F, 0.0F, 0.0F}}; // Clear color
		VkDevice m_device = VK_NULL_HANDLE; // Convenient sortcut to device

		// Display G-Buffers (one per frame-in-flight slot; swapchain uses up to 3).
		nvvk::GBuffer m_gBuffers[3];

		// camera info for current frame, updated by onRender
		glm::vec3 m_eye{};
		glm::vec3 m_center{};
		glm::vec3 m_up{};

		// IndirectParams structure defined in shaderio.h
		nvvk::Buffer m_indirect; // indirect parameter buffer
		nvvk::Buffer m_indirectReadbackHost; // buffer for readback
		shaderio::IndirectParams m_indirectReadback; // readback values
		bool m_canCollectReadback = false; // tells wether readback will be available in Host buffer at next frame

		// TODO maybe move that in SplatSetVK next to icosa, and the associated init/deinit
		nvvk::Buffer m_quadVertices; // Buffer of vertices for the splat quad
		nvvk::Buffer m_quadIndices; // Buffer of indices for the splat quad


		SplatSorterAsync m_cpuSorter; // CPU async sorting
		std::vector<uint32_t> m_splatIndices; // the array of cpu sorted indices to use for rendering
		VrdxSorter m_gpuSorter = VK_NULL_HANDLE; // GPU radix sort

		// buffers used by GPU and/or CPU sort
		nvvk::Buffer m_splatIndicesHost; // Buffer of splat indices on host for transfers (used by CPU sort)
		nvvk::Buffer m_splatIndicesDevice; // Buffer of splat indices on device (used by CPU and GPU sort)
		nvvk::Buffer m_splatDistancesDevice; // Buffer of splat indices on device (used by CPU and GPU sort)
		nvvk::Buffer m_vrdxStorageDevice; // Used internally by VrdxSorter, GPU sort

		// macro definitions shared by all shaders
		std::vector<std::pair<std::string, std::string>> m_shaderMacros;
		// used to load and compile shaders
		nvslang::SlangCompiler m_slangCompiler{};

		// The different shaders that are used in the pipelines
		struct Shaders
		{
			// 3D Gaussians Raster
			VkShaderModule distShader{};
			VkShaderModule meshShader{};
			VkShaderModule vertexShader{};
			VkShaderModule fragmentShader{};
			VkShaderModule threedgutMeshShader{};
			VkShaderModule threedgutFragmentShader{};
			// Utility storage to process shaders in loop
			std::vector<VkShaderModule*> modules{};
			// true if all the shaders are succesfully build
			bool valid = false;
		} m_shaders;

		// 3D Gaussians Pipelines
		VkPipeline m_computePipelineGsDistCull = VK_NULL_HANDLE;
		// The compute pipeline to compute gaussian splats distances to eye and cull
		VkPipeline m_graphicsPipelineGsVert = VK_NULL_HANDLE;
		// The graphic pipeline to rasterize gaussian splats using vertex shaders
		VkPipeline m_graphicsPipelineGsMesh = VK_NULL_HANDLE;
		// The graphic pipeline to rasterize gaussian splats using mesh shaders
		VkPipeline m_graphicsPipeline3dgutMesh = VK_NULL_HANDLE;
		// The graphic pipeline to rasterize 3DGUT splats using mesh shaders
		// Common to 3D Gaussians pipeline
		VkPipelineLayout m_pipelineLayout = VK_NULL_HANDLE; // Raster Pipelines layout
		VkDescriptorSetLayout m_descriptorSetLayout = VK_NULL_HANDLE; // Descriptor set layout
		VkDescriptorSet m_descriptorSet = VK_NULL_HANDLE; // Raster Descriptor set
		VkDescriptorPool m_descriptorPool = VK_NULL_HANDLE; // Raster Descriptor pool

		nvvk::Buffer m_frameInfoBuffer; // uniform buffer to store frame parameters defined by global variable prmFrame

	};
} // namespace vk_gaussian_splatting

#endif
