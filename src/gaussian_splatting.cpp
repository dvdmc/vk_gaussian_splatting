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

// Vulkan Memory Allocator
#define VMA_IMPLEMENTATION
#define VMA_LEAK_LOG_FORMAT(format, ...)                                                                               \
  {                                                                                                                    \
    printf((format), __VA_ARGS__);                                                                                     \
    printf("\n");                                                                                                      \
  }

#include "gaussian_splatting.h"
#include "utilities.h"

#define GLM_ENABLE_SWIZZLE
#include <glm/gtc/packing.hpp>  // Required for half-float operations

#include <nvvk/check_error.hpp>
#include <nvvk/descriptors.hpp>
#include <nvvk/graphics_pipeline.hpp>
#include <nvvk/formats.hpp>


namespace vk_gaussian_splatting
{
	GaussianSplatting::GaussianSplatting()
		: cameraManip(std::make_shared<nvutils::CameraManipulator>())
	{
	};

	GaussianSplatting::~GaussianSplatting()
	{
		// all threads must be stopped,
		// work done in onDetach(),
		// could be done here, same result
	};

	void GaussianSplatting::onAttach(nvapp::Application* app)
	{
		// shortcuts
		m_app = app;
		m_device = m_app->getDevice();

		// starts the asynchronous services
		m_plyLoader.initialize();

		// Memory allocator
		m_alloc.init(VmaAllocatorCreateInfo{
			.flags = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT,
			.physicalDevice = app->getPhysicalDevice(),
			.device = app->getDevice(),
			.instance = app->getInstance(),
			.vulkanApiVersion = VK_API_VERSION_1_4,
		});

		// DEBUG: uncomment and set id to find object leak
		// m_alloc.setLeakID(70);

		// set up buffer uploading utility
		m_uploader.init(&m_alloc, true);

		// Acquiring the sampler which will be used for displaying the GBuffer and accessing textures
		m_samplerPool.init(app->getDevice());
		NVVK_CHECK(m_samplerPool.acquireSampler(m_sampler));
		NVVK_DBG_NAME(m_sampler);

		// GBuffer
		m_depthFormat = nvvk::findDepthFormat(app->getPhysicalDevice());

    for(nvvk::GBuffer& buffer: m_gBuffers)
      buffer.init({
			.allocator = &m_alloc,
			.colorFormats = {m_colorFormat},
			.depthFormat = m_depthFormat,
			.imageSampler = m_sampler,
			.descriptorPool = m_app->getTextureDescriptorPool(),
		});

		// Initialize the splat renderer
		m_splatRenderer.init(m_app, &m_alloc, &m_uploader, &m_sampler, m_colorFormat, m_depthFormat);
		m_cameraSet.init(cameraManip.get());

		// Initialize the GLTF rasterizer
		m_gltfRasterizer.init(m_app, &m_alloc, &m_samplerPool, m_colorFormat, m_depthFormat);

		// Off-screen GBuffer for virtual camera renders.
		m_offscreenGBuffer.init({
			.allocator    = &m_alloc,
			.colorFormats = {m_colorFormat},
			.depthFormat  = m_depthFormat,
			.imageSampler = m_sampler,
		});
		VkCommandBuffer initCmd = m_app->createTempCmdBuffer();
		NVVK_CHECK(m_offscreenGBuffer.update(initCmd, m_virtualCameraResolution));
		m_app->submitAndWaitTempCmdBuffer(initCmd);
	};

	void GaussianSplatting::onDetach()
	{
		// stops the threads
		m_plyLoader.shutdown();
		// release scene and rendering related resources
		deinitAll();
		// release application wide related resources
		m_gltfRasterizer.deinit();
		m_splatRenderer.deinit();
		for(nvvk::GBuffer& gBuffer : m_gBuffers)
			gBuffer.deinit();
		m_offscreenGBuffer.deinit();
		m_samplerPool.releaseSampler(m_sampler);
		m_samplerPool.deinit();
		m_uploader.deinit();

		// Free per-virtual-camera readback buffers
		for(auto& capture : m_cameraCaptures)
		{
			if(capture->buffer.buffer)
				vkDestroyBuffer(m_device, capture->buffer.buffer, nullptr);
			if(capture->buffer.memory)
				vkFreeMemory(m_device, capture->buffer.memory, nullptr);
		}
		m_cameraCaptures.clear();

		m_alloc.deinit();
	}

	void GaussianSplatting::loadGltfScene(const std::filesystem::path& path)
	{
		if(!path.empty() && m_gltfRasterizer.loadGltfScene(path))
			m_loadedGltfFilename = path;
	}

	void GaussianSplatting::loadHdr(const std::filesystem::path& path)
	{
		if(!path.empty() && m_gltfRasterizer.loadHdr(path))
			m_loadedHdrFilename = path;
	}

	void GaussianSplatting::onResize(VkCommandBuffer cmd, const VkExtent2D& viewportSize)
	{
		m_viewSize = {viewportSize.width, viewportSize.height};
		for(nvvk::GBuffer& gBuffer : m_gBuffers)
			NVVK_CHECK(gBuffer.update(cmd, viewportSize));
		// (GLTF rasterizer updates its output image descriptor per-frame in draw())
	}

	void GaussianSplatting::onRender(VkCommandBuffer cmd)
  {
    NVVK_DBG_SCOPE(cmd);

    // Copy previous frame's virtual camera readback buffers into CameraCapture::image
    // (safe here because frame-in-flight sync guarantees the GPU has finished those writes)
    collectVirtualCameraCaptures();

    // update buffers, rebuild shaders and pipelines if needed
    m_splatRenderer.setCameraModel(m_cameraSet.getCamera().model);
    m_splatRenderer.processUpdateRequests(m_splatSet);

    // 0 if not ready so the rendering does not
    // touch the splat set while loading
    // getStatus is thread safe.
    uint32_t splatCount = 0;
    if(m_plyLoader.getStatus() == PlyLoaderAsync::State::E_READY)
    {
      splatCount = (uint32_t)m_splatSet.size();
    }

    ///////////////////
    // From this point we are using full raster or hybrid.

    // Use the GBuffer slot matching the current frame-in-flight index.
    // The swapchain uses up to 3 concurrent frames; each gets its own slot
    // so no two in-flight frames write the same image simultaneously.
    nvvk::GBuffer& gBuffer = m_gBuffers[m_app->getFrameCycleIndex()];

    // Build DrawParams for the display camera
    GaussianSplatRenderer::DrawParams drawParams{};
    {
      Camera camera = m_cameraSet.getCamera();
      glm::vec3 eye, center, up;
      cameraManip->getLookat(eye, center, up);

      drawParams.view         = cameraManip->getViewMatrix();
      drawParams.proj         = cameraManip->getPerspectiveMatrix();
      drawParams.eye          = eye;
      drawParams.center       = center;
      drawParams.up           = up;
      drawParams.fovRad       = cameraManip->getRadFov();
      drawParams.nearFar      = cameraManip->getClipPlanes();
      drawParams.viewportSize = m_app->getViewportSize();
      drawParams.cameraModel  = camera.model;
      drawParams.focusDist    = camera.focusDist;
      drawParams.aperture     = camera.aperture;
      drawParams.hasGltfScene = m_gltfRasterizer.hasScene();
    }

    // Handle device-host data update and splat sorting if a scene exist
    if(m_splatRenderer.isReady() && splatCount)
    {
      // collect readback results from previous frame if any
      m_splatRenderer.collectReadback();

      // upload UBO and sort
      m_splatRenderer.prepareDraw(cmd, drawParams);
    }

    // In which color buffer are we going to render ?
    uint32_t colorBufferId = 0;

    nvvk::cmdImageMemoryBarrier(cmd, {gBuffer.getDepthImage(),
                                      VK_IMAGE_LAYOUT_UNDEFINED,  // discard previous content; safe because depth is always cleared before use
                                      VK_IMAGE_LAYOUT_GENERAL,
                                      {VK_IMAGE_ASPECT_DEPTH_BIT, 0, VK_REMAINING_MIP_LEVELS, 0, VK_REMAINING_ARRAY_LAYERS}});

    // Draw GLTF scene + environment (sky/HDR) before the splat pass.
    // Both images are in GENERAL on entry and will be returned in GENERAL.
    // The splat pass then uses LOAD to composite on top.
    {
      m_gltfRasterizer.draw(cmd, gBuffer.getSize(),
                            gBuffer.getColorImage(colorBufferId), gBuffer.getColorImageView(colorBufferId),
                            gBuffer.getDepthImage(), gBuffer.getDepthImageView(),
                            drawParams.view, drawParams.proj);
    }

    // Drawing the primitives in the G-Buffer
    {
      const VkExtent2D& viewportSize = m_app->getViewportSize();
      const VkViewport  viewport{0.0F, 0.0F, float(viewportSize.width), float(viewportSize.height), 0.0F, 1.0F};
      const VkRect2D    scissor{{0, 0}, viewportSize};

      VkRenderingAttachmentInfo colorAttachment = DEFAULT_VkRenderingAttachmentInfo;
      colorAttachment.imageView                 = gBuffer.getColorImageView(colorBufferId);
      // LOAD: GLTF/sky already rendered into this image; splats composite on top
      colorAttachment.loadOp     = VK_ATTACHMENT_LOAD_OP_LOAD;
      colorAttachment.clearValue = {m_clearColor};
      VkRenderingAttachmentInfo depthAttachment = DEFAULT_VkRenderingAttachmentInfo;
      depthAttachment.imageView  = gBuffer.getDepthImageView();
      depthAttachment.clearValue = {.depthStencil = DEFAULT_VkClearDepthStencilValue};
      // LOAD depth when GLTF scene rendered geometry (so splats respect GLTF occlusion)
      if(m_gltfRasterizer.hasScene())
        depthAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;

      // Create the rendering info
      VkRenderingInfo renderingInfo      = DEFAULT_VkRenderingInfo;
      renderingInfo.renderArea           = DEFAULT_VkRect2D(gBuffer.getSize());
      renderingInfo.colorAttachmentCount = 1;
      renderingInfo.pColorAttachments    = &colorAttachment;
      renderingInfo.pDepthAttachment     = &depthAttachment;

      nvvk::cmdImageMemoryBarrier(cmd, {gBuffer.getColorImage(colorBufferId), VK_IMAGE_LAYOUT_GENERAL,
                                        VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL});

      nvvk::cmdImageMemoryBarrier(cmd, {gBuffer.getDepthImage(),
                                        VK_IMAGE_LAYOUT_GENERAL,
                                        VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
                                        {VK_IMAGE_ASPECT_DEPTH_BIT, 0, VK_REMAINING_MIP_LEVELS, 0, VK_REMAINING_ARRAY_LAYERS}});

      vkCmdBeginRendering(cmd, &renderingInfo);

      vkCmdSetViewportWithCount(cmd, 1, &viewport);
      vkCmdSetScissorWithCount(cmd, 1, &scissor);

      // splat set (OBJ mesh pipeline removed; GLTF rendered above)
      if(m_splatRenderer.isReady() && splatCount)
      {
        m_splatRenderer.drawSplatPrimitives(cmd, drawParams);
      }

      vkCmdEndRendering(cmd);

      nvvk::cmdImageMemoryBarrier(cmd, {gBuffer.getColorImage(colorBufferId),
                                        VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL});
      nvvk::cmdImageMemoryBarrier(cmd, {gBuffer.getDepthImage(),
                                        VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
                                        VK_IMAGE_LAYOUT_GENERAL,
                                        {VK_IMAGE_ASPECT_DEPTH_BIT, 0, VK_REMAINING_MIP_LEVELS, 0, VK_REMAINING_ARRAY_LAYERS}});
    }

    // Render virtual cameras (indices 1..N) into the offscreen GBuffer and readback.
    if(m_splatRenderer.isReady() && splatCount)
      renderVirtualCameras(cmd, splatCount);

		m_splatRenderer.postDraw(cmd);
	}








	void GaussianSplatting::deinitAll()
	{
		vkDeviceWaitIdle(m_device);

		deinitScene();
		m_splatRenderer.unloadSplatData();
		m_cameraSet.deinit();
		resetRenderSettings();
		// record default cam for reset in UI
		m_cameraSet.setCamera(Camera());
		// record default cam for reset in UI
		m_cameraSet.setHomePreset(m_cameraSet.getCamera());
	}

	bool GaussianSplatting::initAll()
	{
		vkDeviceWaitIdle(m_device);

		// TODO: use BBox of point cloud to set far plane, eye and center
		m_cameraSet.setCamera(Camera());
		// record default cam for reset in UI
		m_cameraSet.setHomePreset(m_cameraSet.getCamera());
		// reset general parameters
		resetRenderSettings();

		m_splatRenderer.setCameraModel(m_cameraSet.getCamera().model);
		return m_splatRenderer.loadSplatData(m_splatSet);
	}

	void GaussianSplatting::deinitScene()
	{
		m_splatSet = {};
		m_loadedSceneFilename = "";
	}


	VkResult GaussianSplatting::updateReadbackBuffer(VkDevice device, VkPhysicalDevice physicalDevice, VkDeviceSize size, ReadbackBuffer& out)
	{
		if (size == out.size)
	      return VK_SUCCESS;  // Nothing to do

		out = {};
		out.size = size;

		VkBufferCreateInfo bufferInfo{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
		bufferInfo.size = size;
		bufferInfo.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
		bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

		VkResult res = vkCreateBuffer(device, &bufferInfo, nullptr, &out.buffer);
		if (res != VK_SUCCESS)
			throw std::runtime_error("vkCreateBuffer failed for readback buffer");

		VkMemoryRequirements memReq{};
		vkGetBufferMemoryRequirements(device, out.buffer, &memReq);

		VkPhysicalDeviceMemoryProperties memProps{};
		vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProps);

		uint32_t memoryTypeIndex = UINT32_MAX;
		for (uint32_t i = 0; i < memProps.memoryTypeCount; i++)
		{
			const bool supported = (memReq.memoryTypeBits & (1u << i)) != 0;
			const bool wanted =
				(memProps.memoryTypes[i].propertyFlags & (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
					VK_MEMORY_PROPERTY_HOST_COHERENT_BIT))
				== (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

			if (supported && wanted)
			{
				memoryTypeIndex = i;
				break;
			}
		}

		if (memoryTypeIndex == UINT32_MAX)
			throw std::runtime_error("No host-visible coherent memory type found for readback buffer");

		VkMemoryAllocateInfo allocInfo{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
		allocInfo.allocationSize = memReq.size;
		allocInfo.memoryTypeIndex = memoryTypeIndex;

		res = vkAllocateMemory(device, &allocInfo, nullptr, &out.memory);
		if (res != VK_SUCCESS)
		{
			vkDestroyBuffer(device, out.buffer, nullptr);
			out.buffer = VK_NULL_HANDLE;
			throw std::runtime_error("vkAllocateMemory failed for readback buffer");
		}

		res = vkBindBufferMemory(device, out.buffer, out.memory, 0);
		if (res != VK_SUCCESS)
		{
			vkFreeMemory(device, out.memory, nullptr);
			vkDestroyBuffer(device, out.buffer, nullptr);
			out = {};
			throw std::runtime_error("vkBindBufferMemory failed for readback buffer");
		}

		return VK_SUCCESS;
	}

	// ---------------------------------------------------------------------------
	// Multi-camera interface
	// ---------------------------------------------------------------------------

	uint64_t GaussianSplatting::addCamera(const Camera& cam)
	{
		uint64_t index = m_cameraSet.createPreset(cam);
		// Ensure m_cameraCaptures has one entry per virtual camera (preset indices 1..N).
		size_t needed = m_cameraSet.size() - 1;
		while(m_cameraCaptures.size() < needed)
		{
			m_cameraCaptures.push_back(std::make_unique<CameraCapture>());
			VkDeviceSize bufSize = VkDeviceSize(m_virtualCameraResolution.width)
			                       * m_virtualCameraResolution.height * 4;
			if(bufSize > 0 && m_device != VK_NULL_HANDLE)
				updateReadbackBuffer(m_device, m_app->getPhysicalDevice(), bufSize,
				                     m_cameraCaptures.back()->buffer);
		}
		return index;
	}

	bool GaussianSplatting::removeCamera(uint64_t index)
	{
		if(!m_cameraSet.erasePreset(index))
			return false;
		size_t captureIdx = index - 1;
		if(captureIdx < m_cameraCaptures.size())
		{
			auto& cap = *m_cameraCaptures[captureIdx];
			if(cap.buffer.buffer)
				vkDestroyBuffer(m_device, cap.buffer.buffer, nullptr);
			if(cap.buffer.memory)
				vkFreeMemory(m_device, cap.buffer.memory, nullptr);
			m_cameraCaptures.erase(m_cameraCaptures.begin() + captureIdx);
		}
		return true;
	}

	bool GaussianSplatting::setCamera(uint64_t index, const Camera& cam)
	{
		return m_cameraSet.setPreset(index, cam);
	}

	Camera GaussianSplatting::getCamera(uint64_t index) const
	{
		return m_cameraSet.getPreset(index);
	}

	uint64_t GaussianSplatting::getCameraCount() const
	{
		return m_cameraSet.size();
	}

	bool GaussianSplatting::getImage(uint64_t index, std::vector<unsigned char>& out)
	{
		if(index == 0 || index > m_cameraCaptures.size())
			return false;
		auto& capture = *m_cameraCaptures[index - 1];
		std::lock_guard<std::mutex> lock(capture.mutex);
		if(!capture.ready)
			return false;
		out = capture.image;
		return true;
	}

	void GaussianSplatting::setVirtualCameraResolution(VkExtent2D size)
	{
		if(size.width == m_virtualCameraResolution.width
		   && size.height == m_virtualCameraResolution.height)
			return;

		vkDeviceWaitIdle(m_device);
		m_virtualCameraResolution = size;

		VkCommandBuffer cmd = m_app->createTempCmdBuffer();
		NVVK_CHECK(m_offscreenGBuffer.update(cmd, size));
		m_app->submitAndWaitTempCmdBuffer(cmd);

		VkDeviceSize bufSize = VkDeviceSize(size.width) * size.height * 4;
		for(auto& capture : m_cameraCaptures)
		{
			if(capture->buffer.buffer)
				vkDestroyBuffer(m_device, capture->buffer.buffer, nullptr);
			if(capture->buffer.memory)
				vkFreeMemory(m_device, capture->buffer.memory, nullptr);
			capture->buffer = {};
			if(bufSize > 0)
				updateReadbackBuffer(m_device, m_app->getPhysicalDevice(), bufSize,
				                     capture->buffer);
		}
	}

	// ---------------------------------------------------------------------------
	// Private multi-camera helpers
	// ---------------------------------------------------------------------------

	void GaussianSplatting::collectVirtualCameraCaptures()
	{
		for(auto& capture : m_cameraCaptures)
		{
			if(capture->buffer.buffer == VK_NULL_HANDLE || capture->buffer.size == 0)
				continue;
			void* ptr = nullptr;
			vkMapMemory(m_device, capture->buffer.memory, 0, capture->buffer.size, 0, &ptr);
			if(ptr)
			{
				std::lock_guard<std::mutex> lock(capture->mutex);
				capture->image.resize(capture->buffer.size);
				memcpy(capture->image.data(), ptr, capture->buffer.size);
				capture->ready = true;
				vkUnmapMemory(m_device, capture->buffer.memory);
			}
		}
	}

	void GaussianSplatting::uploadFrameInfoForCamera(VkCommandBuffer cmd, uint32_t splatCount,
	                                                  const Camera& cam, VkExtent2D viewport)
	{
		const float     fovRad = glm::radians(cam.fov);
		const float     aspect = float(viewport.width) / float(viewport.height);
		const glm::mat4 view   = glm::lookAt(cam.eye, cam.ctr, cam.up);
		const glm::mat4 proj   = glm::perspective(fovRad, aspect, cam.clip.x, cam.clip.y);

		GaussianSplatRenderer::DrawParams params{};
		params.view         = view;
		params.proj         = proj;
		params.eye          = cam.eye;
		params.center       = cam.ctr;
		params.up           = cam.up;
		params.fovRad       = fovRad;
		params.nearFar      = cam.clip;
		params.viewportSize = viewport;
		params.cameraModel  = cam.model;
		params.focusDist    = cam.focusDist;
		params.aperture     = cam.aperture;
		params.hasGltfScene = m_gltfRasterizer.hasScene();

		m_splatRenderer.prepareDraw(cmd, params);
	}

	void GaussianSplatting::renderVirtualCameras(VkCommandBuffer cmd, uint32_t splatCount)
	{
		if(m_cameraCaptures.empty())
			return;

		// Virtual cameras always use GPU sort. Warn once if the user had CPU sort selected.
		if(prmRaster.sortingMethod != SORTING_GPU_SYNC_RADIX)
			LOGW("Virtual cameras require GPU sort; CPU sort will not be used for virtual camera passes.\n");

		const uint32_t colorId = 0;

		for(size_t i = 0; i < m_cameraCaptures.size(); i++)
		{
			// preset index i+1 (preset[0] is the interactive display camera)
			const Camera cam = m_cameraSet.getPreset(uint64_t(i + 1));

			// Build DrawParams for this camera
			const float     fovRad = glm::radians(cam.fov);
			const float     aspect = float(m_virtualCameraResolution.width)
			                         / float(m_virtualCameraResolution.height);
			const glm::mat4 view   = glm::lookAt(cam.eye, cam.ctr, cam.up);
			const glm::mat4 proj   = glm::perspective(fovRad, aspect, cam.clip.x, cam.clip.y);

			GaussianSplatRenderer::DrawParams drawParams{};
			drawParams.view         = view;
			drawParams.proj         = proj;
			drawParams.eye          = cam.eye;
			drawParams.center       = cam.ctr;
			drawParams.up           = cam.up;
			drawParams.fovRad       = fovRad;
			drawParams.nearFar      = cam.clip;
			drawParams.viewportSize = m_virtualCameraResolution;
			drawParams.cameraModel  = cam.model;
			drawParams.focusDist    = cam.focusDist;
			drawParams.aperture     = cam.aperture;
			drawParams.hasGltfScene = m_gltfRasterizer.hasScene();

			// 1. Upload frame UBO and sort for this camera
			m_splatRenderer.prepareDraw(cmd, drawParams);

			// 2. Discard previous depth; both images start in GENERAL
			nvvk::cmdImageMemoryBarrier(cmd, {m_offscreenGBuffer.getDepthImage(),
			                                  VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
			                                  {VK_IMAGE_ASPECT_DEPTH_BIT, 0, VK_REMAINING_MIP_LEVELS, 0, VK_REMAINING_ARRAY_LAYERS}});

			// 3. GLTF pass (sky/HDR + mesh geometry) — enters/exits GENERAL
			m_gltfRasterizer.draw(cmd, m_virtualCameraResolution,
			                      m_offscreenGBuffer.getColorImage(colorId),
			                      m_offscreenGBuffer.getColorImageView(colorId),
			                      m_offscreenGBuffer.getDepthImage(),
			                      m_offscreenGBuffer.getDepthImageView(),
			                      view, proj);

			// 4. Splat pass — LOAD over GLTF result
			{
				nvvk::cmdImageMemoryBarrier(cmd, {m_offscreenGBuffer.getColorImage(colorId),
				                                  VK_IMAGE_LAYOUT_GENERAL,
				                                  VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL});
				nvvk::cmdImageMemoryBarrier(cmd, {m_offscreenGBuffer.getDepthImage(),
				                                  VK_IMAGE_LAYOUT_GENERAL,
				                                  VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
				                                  {VK_IMAGE_ASPECT_DEPTH_BIT, 0, VK_REMAINING_MIP_LEVELS, 0, VK_REMAINING_ARRAY_LAYERS}});

				VkRenderingAttachmentInfo colorAttachment = DEFAULT_VkRenderingAttachmentInfo;
				colorAttachment.imageView                 = m_offscreenGBuffer.getColorImageView(colorId);
				colorAttachment.loadOp                    = VK_ATTACHMENT_LOAD_OP_LOAD;

				VkRenderingAttachmentInfo depthAttachment = DEFAULT_VkRenderingAttachmentInfo;
				depthAttachment.imageView                 = m_offscreenGBuffer.getDepthImageView();
				depthAttachment.clearValue                = {.depthStencil = DEFAULT_VkClearDepthStencilValue};
				if(m_gltfRasterizer.hasScene())
					depthAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;

				VkRenderingInfo renderingInfo      = DEFAULT_VkRenderingInfo;
				renderingInfo.renderArea           = DEFAULT_VkRect2D(m_virtualCameraResolution);
				renderingInfo.colorAttachmentCount = 1;
				renderingInfo.pColorAttachments    = &colorAttachment;
				renderingInfo.pDepthAttachment     = &depthAttachment;

				vkCmdBeginRendering(cmd, &renderingInfo);

				const VkViewport viewport{0.f, 0.f, float(m_virtualCameraResolution.width),
				                          float(m_virtualCameraResolution.height), 0.f, 1.f};
				const VkRect2D scissor{{0, 0}, m_virtualCameraResolution};
				vkCmdSetViewportWithCount(cmd, 1, &viewport);
				vkCmdSetScissorWithCount(cmd, 1, &scissor);

				m_splatRenderer.drawSplatPrimitives(cmd, drawParams);
				vkCmdEndRendering(cmd);
			}

			// 5. Readback: COLOR_ATTACHMENT → TRANSFER_SRC, copy, → GENERAL
			nvvk::cmdImageMemoryBarrier(cmd, {m_offscreenGBuffer.getColorImage(colorId),
			                                  VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
			                                  VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL});

			CameraCapture& capture = *m_cameraCaptures[i];
			if(capture.buffer.buffer != VK_NULL_HANDLE)
			{
				VkBufferImageCopy region{};
				region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
				region.imageSubresource.layerCount = 1;
				region.imageExtent = {m_virtualCameraResolution.width,
				                      m_virtualCameraResolution.height, 1};
				vkCmdCopyImageToBuffer(cmd, m_offscreenGBuffer.getColorImage(colorId),
				                       VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
				                       capture.buffer.buffer, 1, &region);
			}

			// Depth still in DEPTH_ATTACHMENT — transition back to GENERAL
			nvvk::cmdImageMemoryBarrier(cmd, {m_offscreenGBuffer.getColorImage(colorId),
			                                  VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
			                                  VK_IMAGE_LAYOUT_GENERAL});
			nvvk::cmdImageMemoryBarrier(cmd, {m_offscreenGBuffer.getDepthImage(),
			                                  VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
			                                  VK_IMAGE_LAYOUT_GENERAL,
			                                  {VK_IMAGE_ASPECT_DEPTH_BIT, 0, VK_REMAINING_MIP_LEVELS, 0, VK_REMAINING_ARRAY_LAYERS}});
		}
	}

} // namespace vk_gaussian_splatting
