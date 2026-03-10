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

#include "nvutils/file_operations.hpp"

#include "nvgui/fonts.hpp"
#include "nvgui/tooltip.hpp"

#include <glm/vec2.hpp>
// clang-format off
#define IM_VEC2_CLASS_EXTRA ImVec2(const glm::vec2& f) {x = f.x; y = f.y;} operator glm::vec2() const { return glm::vec2(x, y); }
// clang-format on

#include <chrono>
#include <thread>
#include <filesystem>
#include <algorithm>  // for std::clamp

#include "gaussian_splatting_ui.h"

namespace vk_gaussian_splatting {

GaussianSplattingUI::GaussianSplattingUI()
    : GaussianSplatting()
{
};

GaussianSplattingUI::~GaussianSplattingUI() {
  // Nothing to do here
};

void GaussianSplattingUI::onAttach(nvapp::Application* app)
{
  // Initializes the core

  GaussianSplatting::onAttach(app);

  // Init combo selectors used in UI

  m_ui.enumAdd(GUI_STORAGE, STORAGE_BUFFERS, "Buffers");
  m_ui.enumAdd(GUI_STORAGE, STORAGE_TEXTURES, "Textures");

  m_ui.enumAdd(GUI_PIPELINE, PIPELINE_VERT, "Raster vertex shader 3DGS");
  m_ui.enumAdd(GUI_PIPELINE, PIPELINE_MESH, "Raster mesh shader 3DGS");
  m_ui.enumAdd(GUI_PIPELINE, PIPELINE_MESH_3DGUT, "Raster mesh shader 3DGUT");

  m_ui.enumAdd(GUI_EXTENT_METHOD, EXTENT_EIGEN, "Eigen");
  m_ui.enumAdd(GUI_EXTENT_METHOD, EXTENT_CONIC, "Conic");

  m_ui.enumAdd(GUI_VISUALIZE, VISUALIZE_FINAL, "Final render");
  m_ui.enumAdd(GUI_VISUALIZE, VISUALIZE_CLOCK, "Clock cycles");
  m_ui.enumAdd(GUI_VISUALIZE, VISUALIZE_DEPTH, "Splats depth");
  m_ui.enumAdd(GUI_VISUALIZE, VISUALIZE_RAYHITS, "Ray Hit Count");

  m_ui.enumAdd(GUI_SORTING, SORTING_GPU_SYNC_RADIX, "GPU radix sort");
  m_ui.enumAdd(GUI_SORTING, SORTING_CPU_ASYNC_MULTI, "CPU async std multi");

  m_ui.enumAdd(GUI_SH_FORMAT, FORMAT_FLOAT32, "Float 32");
  m_ui.enumAdd(GUI_SH_FORMAT, FORMAT_FLOAT16, "Float 16");
  m_ui.enumAdd(GUI_SH_FORMAT, FORMAT_UINT8, "Uint8");

  m_ui.enumAdd(GUI_PARTICLE_FORMAT, PARTICLE_FORMAT_ICOSAHEDRON, "Icosahedron");
  m_ui.enumAdd(GUI_PARTICLE_FORMAT, PARTICLE_FORMAT_PARAMETRIC, "AABB + parametric");

  m_ui.enumAdd(GUI_CAMERA_TYPE, CAMERA_PINHOLE, "Pinhole");
  m_ui.enumAdd(GUI_CAMERA_TYPE, CAMERA_FISHEYE, "Fisheye");

  m_ui.enumAdd(GUI_TEMPORAL_SAMPLING, TEMPORAL_SAMPLING_AUTO, "Automatic");
  m_ui.enumAdd(GUI_TEMPORAL_SAMPLING, TEMPORAL_SAMPLING_ENABLED, "Force enabled");
  m_ui.enumAdd(GUI_TEMPORAL_SAMPLING, TEMPORAL_SAMPLING_DISABLED, "Force disabled");

  m_ui.enumAdd(GUI_KERNEL_DEGREE, KERNEL_DEGREE_QUINTIC, "5 (Quintic)");
  m_ui.enumAdd(GUI_KERNEL_DEGREE, KERNEL_DEGREE_TESSERACTIC, "4 (Tesseractic)");
  m_ui.enumAdd(GUI_KERNEL_DEGREE, KERNEL_DEGREE_CUBIC, "3 (Cubic)");
  m_ui.enumAdd(GUI_KERNEL_DEGREE, KERNEL_DEGREE_QUADRATIC, "2 (Quadratic)");
  m_ui.enumAdd(GUI_KERNEL_DEGREE, KERNEL_DEGREE_LAPLACIAN, "1 (Laplacian)");
  m_ui.enumAdd(GUI_KERNEL_DEGREE, KERNEL_DEGREE_LINEAR, "0 (Linear)");

  m_ui.enumAdd(GUI_LIGHT_TYPE, LIGHT_TYPE_POINT, "Point");
  m_ui.enumAdd(GUI_LIGHT_TYPE, LIGHT_TYPE_DIRECTIONAL, "Directional");

  m_ui.enumAdd(GUI_ILLUM_MODEL, 0, "No indirect");
  m_ui.enumAdd(GUI_ILLUM_MODEL, 1, "Reflective");
  m_ui.enumAdd(GUI_ILLUM_MODEL, 2, "Refractive");

  m_ui.enumAdd(GUI_DIST_SHADER_WG_SIZE, 512, "512");
  m_ui.enumAdd(GUI_DIST_SHADER_WG_SIZE, 256, "256");
  m_ui.enumAdd(GUI_DIST_SHADER_WG_SIZE, 128, "128");
  m_ui.enumAdd(GUI_DIST_SHADER_WG_SIZE, 64, "64");
  m_ui.enumAdd(GUI_DIST_SHADER_WG_SIZE, 32, "32");
  m_ui.enumAdd(GUI_DIST_SHADER_WG_SIZE, 16, "16");

  m_ui.enumAdd(GUI_MESH_SHADER_WG_SIZE, 128, "128");
  m_ui.enumAdd(GUI_MESH_SHADER_WG_SIZE, 64, "64");
  m_ui.enumAdd(GUI_MESH_SHADER_WG_SIZE, 32, "32");
  m_ui.enumAdd(GUI_MESH_SHADER_WG_SIZE, 16, "16");
  m_ui.enumAdd(GUI_MESH_SHADER_WG_SIZE, 8, "8");

  m_ui.enumAdd(GUI_RAY_HIT_PER_PASS, 128, "128");
  m_ui.enumAdd(GUI_RAY_HIT_PER_PASS, 64, "64");
  m_ui.enumAdd(GUI_RAY_HIT_PER_PASS, 32, "32");
  m_ui.enumAdd(GUI_RAY_HIT_PER_PASS, 20, "20");
  m_ui.enumAdd(GUI_RAY_HIT_PER_PASS, 18, "18");
  m_ui.enumAdd(GUI_RAY_HIT_PER_PASS, 16, "16");
  m_ui.enumAdd(GUI_RAY_HIT_PER_PASS, 8, "8");
  m_ui.enumAdd(GUI_RAY_HIT_PER_PASS, 4, "4");
}

void GaussianSplattingUI::onDetach()
{
  GaussianSplatting::onDetach();
}

void GaussianSplattingUI::onResize(VkCommandBuffer cmd, const VkExtent2D& size)
{
  GaussianSplatting::onResize(cmd, size);
}

void GaussianSplattingUI::onPreRender()
{
  GaussianSplatting::onPreRender();
}

void GaussianSplattingUI::onRender(VkCommandBuffer cmd)
{
  GaussianSplatting::onRender(cmd);
}

#define ICON_BLANK "     "

void GaussianSplattingUI::onUIRender()
{
  /////////////
  // Rendering Viewport display the GBuffer
  {
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0F, 0.0F));
    ImGui::Begin("Viewport");

    // Display the G-Buffer image
    ImGui::Image((ImTextureID)m_gBuffers.getDescriptorSet(), ImGui::GetContentRegionAvail());

    ImVec2 wp = ImGui::GetWindowPos();
    ImVec2 ws = ImGui::GetWindowSize();

    // display the basis widget at bottom left
    float  size   = 25.F;
    ImVec2 offset = ImVec2(size * 1.1F, -size * 1.1F) * ImGui::GetWindowDpiScale();
    ImVec2 pos    = ImVec2(wp.x, wp.y + ws.y) + offset;
    nvgui::Axis(pos, cameraManip->getViewMatrix(), size);

    // store mouse cursor
    // will be available for next frame in frameInfo
    ImVec2 mp = ImGui::GetMousePos();  // Mouse position in screen space

    // Convert to viewport space (0,0 at bottom-left)
    ImVec2 mouseInViewport = ImVec2(mp.x - wp.x, mp.y - wp.y);

    if(mouseInViewport.x < 0 || mouseInViewport.y < 0 || mouseInViewport.x >= ws.x || mouseInViewport.y >= ws.y)
      prmFrame.cursor.x = prmFrame.cursor.y = -1;  // just so it is easy to test in shader if pos is valid
    else
      prmFrame.cursor = {mouseInViewport.x, mouseInViewport.y};

    ImGui::End();
    ImGui::PopStyleVar();
  }


  /////////////////
  // Handle scene loading

#ifdef WITH_DEFAULT_SCENE_FEATURE
  // load a default scene if none was provided by command line
  if(prmScene.enableDefaultScene && m_loadedSceneFilename.empty() && prmScene.sceneToLoadFilename.empty()
     && m_plyLoader.getStatus() == PlyLoaderAsync::State::E_READY)
  {
    const std::vector<std::filesystem::path> defaultSearchPaths = getResourcesDirs();
    prmScene.sceneToLoadFilename = nvutils::findFile("flowers_1/flowers_1.ply", defaultSearchPaths).string();
    prmScene.enableDefaultScene  = false;
  }
#endif

  // do we need to load a new scene ?
  if(!prmScene.sceneToLoadFilename.empty() && m_plyLoader.getStatus() == PlyLoaderAsync::State::E_READY)
  {

    if(!m_loadedSceneFilename.empty() && prmScene.projectToLoadFilename.empty())
      ImGui::OpenPopup("Load .ply file ?");

    // Always center this window when appearing
    ImVec2 center = ImGui::GetMainViewport()->GetCenter();
    ImGui::SetNextWindowPos(center, ImGuiCond_Appearing, ImVec2(0.5f, 0.5f));

    bool doReset = true;

    if(ImGui::BeginPopupModal("Load .ply file ?", NULL, ImGuiWindowFlags_AlwaysAutoResize))
    {
      doReset = false;

      ImGui::Text("The current project will be entirely replaced.\nThis operation cannot be undone!");
      ImGui::Separator();

      if(ImGui::Button("OK", ImVec2(120, 0)))
      {
        doReset = true;
        ImGui::CloseCurrentPopup();
      }
      ImGui::SetItemDefaultFocus();
      ImGui::SameLine();
      if(ImGui::Button("Cancel", ImVec2(120, 0)))
      {
        // cancel any request leading to a reset
        prmScene.sceneToLoadFilename   = "";
        prmScene.projectToLoadFilename = "";
        ImGui::CloseCurrentPopup();
      }
      ImGui::EndPopup();
    }

    if(doReset)
    {
      // reset if a scene already exists
      const auto splatCount = m_splatSet.positions.size() / 3;
      if(splatCount)
      {
        deinitAll();
      }

      m_loadedSceneFilename = prmScene.sceneToLoadFilename;
      //
      vkDeviceWaitIdle(m_device);

      std::cout << "Start loading file " << prmScene.sceneToLoadFilename << std::endl;
      if(!m_plyLoader.loadScene(prmScene.sceneToLoadFilename, m_splatSet))
      {
        // this should never occur since status is READY.
        std::cout << "Error: cannot start scene load while loader is not ready status=" << m_plyLoader.getStatus() << std::endl;
      }
      else
      {
        // open the modal window that will collect results
        ImGui::OpenPopup("Loading");
      }

      // reset request
      prmScene.sceneToLoadFilename.clear();
    }
  }

  // display loading jauge modal window
  // Always center this window when appearing
  ImVec2 center = ImGui::GetMainViewport()->GetCenter();
  ImGui::SetNextWindowPos(center, ImGuiCond_Appearing, ImVec2(0.5f, 0.5f));
  if(ImGui::BeginPopupModal("Loading", NULL, ImGuiWindowFlags_AlwaysAutoResize))
  {
    // managment of async load
    switch(m_plyLoader.getStatus())
    {
      case PlyLoaderAsync::State::E_LOADING: {
        ImGui::Text("%s", m_plyLoader.getFilename().string().c_str());
        ImGui::ProgressBar(m_plyLoader.getProgress(), ImVec2(ImGui::GetContentRegionAvail().x, 0.0f));
      }
      break;
      case PlyLoaderAsync::State::E_FAILURE: {
        ImGui::Text("Error: invalid ply file");
        if(ImGui::Button("Ok", ImVec2(120, 0)))
        {
          m_loadedSceneFilename = "";
          // destroy scene just in case it was
          // loaded but not properly since in error
          deinitScene();
          // set ready for next load
          m_plyLoader.reset();
          ImGui::CloseCurrentPopup();
        }
      }
      break;
      case PlyLoaderAsync::State::E_LOADED: {
        // TODO add error modal or better continue on error since it is false only if shaders does not compile
        // Then print shader compilation error directly as a viewport overlay
        // Will allow for fix and hot reload
        if(!initAll())
        {
          // destroy scene
          deinitScene();
        }
        else
        {
          guiAddToRecentFiles(m_loadedSceneFilename);
        }
        // set ready for next load
        m_plyLoader.reset();
        ImGui::CloseCurrentPopup();
      }
      break;
      default: {
        // nothing to do for READY or SHUTDOWN
      }
    }
    ImGui::EndPopup();
  }
}

bool GaussianSplattingUI::guiGetTransform(glm::vec3& scale,
                                          glm::vec3& rotation,
                                          glm::vec3& translation,
                                          glm::mat4& transform,
                                          glm::mat4& transformInv,
                                          bool       disabled /*=false*/)
{
  namespace PE = nvgui::PropertyEditor;

  bool updated = false;
  ImGui::BeginDisabled(disabled);
  updated |= PE::DragFloat3("Translate", glm::value_ptr(translation), 0.05f);
  updated |= PE::DragFloat3("Rotate", glm::value_ptr(rotation), 0.5f);
  updated |= PE::DragFloat3("Scale", glm::value_ptr(scale), 0.01f);
  ImGui::EndDisabled();

  if(updated)
  {
    computeTransform(scale, rotation, translation, transform, transformInv);
  }

  return updated;
}

void GaussianSplattingUI::guiAddToRecentFiles(std::filesystem::path filePath, int historySize)
{
  // first check if filePath is absolute
  if(filePath.is_relative())
  {
    filePath = std::filesystem::absolute(filePath);
  }
  //
  auto it = std::find(m_recentFiles.begin(), m_recentFiles.end(), filePath);
  if(it != m_recentFiles.end())
  {
    m_recentFiles.erase(it);
  }
  m_recentFiles.insert(m_recentFiles.begin(), filePath);
  if(m_recentFiles.size() > historySize)
  {
    m_recentFiles.pop_back();
  }
}

void GaussianSplattingUI::guiAddToRecentProjects(std::filesystem::path filePath, int historySize)
{
  // first check if filePath is absolute
  if(filePath.is_relative())
  {
    filePath = std::filesystem::absolute(filePath);
  }
  //
  auto it = std::find(m_recentProjects.begin(), m_recentProjects.end(), filePath);
  if(it != m_recentProjects.end())
  {
    m_recentProjects.erase(it);
  }
  m_recentProjects.insert(m_recentProjects.begin(), filePath);
  if(m_recentProjects.size() > historySize)
  {
    m_recentProjects.pop_back();
  }
}

void GaussianSplattingUI::guiRegisterIniFileHandlers()
{
  // mandatory to work, see ImGui::DockContextInitialize as an example
  auto readOpen = [](ImGuiContext*, ImGuiSettingsHandler* handler, const char* name) -> void* {
    if(strcmp(name, "Data") != 0)
      return NULL;
    // Make sure we clear out our current recent vectors so we don't just keep adding to the list every time we load
    // This is if the .ini file is loaded twice, which happens in nvpro_core2
    auto* ui = static_cast<GaussianSplattingUI*>(handler->UserData);
    if(strcmp(handler->TypeName, "RecentFiles") == 0)
    {
      ui->m_recentFiles.clear();
    }
    else if(strcmp(handler->TypeName, "RecentProjects") == 0)
    {
      ui->m_recentProjects.clear();
    }
    return (void*)1;
  };

  {
    // Save settings handler, not using capture so can be used as a function pointer
    auto saveRecentFilesToIni = [](ImGuiContext* ctx, ImGuiSettingsHandler* handler, ImGuiTextBuffer* buf) {
      auto* self = static_cast<GaussianSplattingUI*>(handler->UserData);
      buf->appendf("[%s][Data]\n", handler->TypeName);
      for(const auto& file : self->m_recentFiles)
      {
        buf->appendf("File=%s\n", file.string().c_str());
      }
      buf->append("\n");
    };

    // Load settings handler, not using capture so can be used as a function pointer
    auto loadRecentFilesFromIni = [](ImGuiContext* ctx, ImGuiSettingsHandler* handler, void* entry, const char* line) {
      auto* self = static_cast<GaussianSplattingUI*>(handler->UserData);
      if(strncmp(line, "File=", 5) == 0)
      {
        const char* filePath = line + 5;
        self->m_recentFiles.push_back(filePath);
      }
    };

    //
    ImGuiSettingsHandler iniHandler;
    iniHandler.TypeName   = "RecentFiles";
    iniHandler.TypeHash   = ImHashStr(iniHandler.TypeName);
    iniHandler.ReadOpenFn = readOpen;
    iniHandler.WriteAllFn = saveRecentFilesToIni;
    iniHandler.ReadLineFn = loadRecentFilesFromIni;
    iniHandler.UserData   = this;  // Pass the current instance to the handler
    ImGui::GetCurrentContext()->SettingsHandlers.push_back(iniHandler);
  }
  {
    // Save settings handler, not using capture so can be used as a function pointer
    auto saveRecentProjectsToIni = [](ImGuiContext* ctx, ImGuiSettingsHandler* handler, ImGuiTextBuffer* buf) {
      auto* self = static_cast<GaussianSplattingUI*>(handler->UserData);
      buf->appendf("[%s][Data]\n", handler->TypeName);
      for(const auto& file : self->m_recentProjects)
      {
        buf->appendf("File=%s\n", file.string().c_str());
      }
      buf->append("\n");
    };

    // Load settings handler, not using capture so can be used as a function pointer
    auto loadRecentProjectsFromIni = [](ImGuiContext* ctx, ImGuiSettingsHandler* handler, void* entry, const char* line) {
      auto* self = static_cast<GaussianSplattingUI*>(handler->UserData);
      if(strncmp(line, "File=", 5) == 0)
      {
        const char* filePath = line + 5;
        self->m_recentProjects.push_back(filePath);
      }
    };

    //
    ImGuiSettingsHandler iniHandler;
    iniHandler.TypeName   = "RecentProjects";
    iniHandler.TypeHash   = ImHashStr(iniHandler.TypeName);
    iniHandler.ReadOpenFn = readOpen;
    iniHandler.WriteAllFn = saveRecentProjectsToIni;
    iniHandler.ReadLineFn = loadRecentProjectsFromIni;
    iniHandler.UserData   = this;  // Pass the current instance to the handler
    ImGui::GetCurrentContext()->SettingsHandlers.push_back(iniHandler);
  }
  {
    // Save window visibility settings handler
    auto saveWindowStatesToIni = [](ImGuiContext* ctx, ImGuiSettingsHandler* handler, ImGuiTextBuffer* buf) {
      auto* self = static_cast<GaussianSplattingUI*>(handler->UserData);
      buf->appendf("[%s][Data]\n", handler->TypeName);
      buf->appendf("ShaderDebugging=%d\n", self->m_showShaderDebugging ? 1 : 0);
      buf->appendf("MemoryStatistics=%d\n", self->m_showMemoryStatistics ? 1 : 0);
      buf->appendf("RendererStatistics=%d\n", self->m_showRendererStatistics ? 1 : 0);
      buf->append("\n");
    };

    // Load window visibility settings handler
    auto loadWindowStatesFromIni = [](ImGuiContext* ctx, ImGuiSettingsHandler* handler, void* entry, const char* line) {
      auto* self = static_cast<GaussianSplattingUI*>(handler->UserData);
      int   value;
#ifdef _MSC_VER
      if(sscanf_s(line, "ShaderDebugging=%d", &value) == 1)
#else
      if(sscanf(line, "ShaderDebugging=%d", &value) == 1)
#endif
      {
        self->m_showShaderDebugging = (value == 1);
      }
#ifdef _MSC_VER
      else if(sscanf_s(line, "MemoryStatistics=%d", &value) == 1)
#else
      else if(sscanf(line, "MemoryStatistics=%d", &value) == 1)
#endif
      {
        self->m_showMemoryStatistics = (value == 1);
      }
#ifdef _MSC_VER
      else if(sscanf_s(line, "RendererStatistics=%d", &value) == 1)
#else
      else if(sscanf(line, "RendererStatistics=%d", &value) == 1)
#endif
      {
        self->m_showRendererStatistics = (value == 1);
      }
    };

    // Custom readOpen for WindowStates that checks for "Data" section
    auto readOpenWindowStates = [](ImGuiContext*, ImGuiSettingsHandler* handler, const char* name) -> void* {
      if(strcmp(name, "Data") != 0)
        return NULL;
      return (void*)1;
    };

    //
    ImGuiSettingsHandler iniHandler;
    iniHandler.TypeName   = "WindowStates";
    iniHandler.TypeHash   = ImHashStr(iniHandler.TypeName);
    iniHandler.ReadOpenFn = readOpenWindowStates;
    iniHandler.WriteAllFn = saveWindowStatesToIni;
    iniHandler.ReadLineFn = loadWindowStatesFromIni;
    iniHandler.UserData   = this;  // Pass the current instance to the handler
    ImGui::GetCurrentContext()->SettingsHandlers.push_back(iniHandler);
  }
}

///////////////////////////////////
// Loading and Saving Propjects

namespace fs = std::filesystem;

fs::path getRelativePath(const fs::path& from, const fs::path& to)
{
  fs::path relativePath;

  auto fromIter = from.begin();
  auto toIter   = to.begin();

  // Find common point
  while(fromIter != from.end() && toIter != to.end() && (*fromIter) == (*toIter))
  {
    ++fromIter;
    ++toIter;
  }

  // Add ".." for each remaining part in `from` path
  for(; fromIter != from.end(); ++fromIter)
  {
    relativePath /= "..";
  }

  // Add remaining part of `to` path
  for(; toIter != to.end(); ++toIter)
  {
    relativePath /= *toIter;
  }

  return relativePath;
}


}  // namespace vk_gaussian_splatting
