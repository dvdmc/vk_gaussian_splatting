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
{};

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

void GaussianSplattingUI::onRender(VkCommandBuffer cmd)
{
  GaussianSplatting::onRender(cmd);
}

#define ICON_BLANK "     "

void GaussianSplattingUI::onUIMenu()
{
  static bool close_app{false};
  bool        v_sync = m_app->isVsync();
#ifndef NDEBUG
  static bool s_showDemo{false};
  static bool s_showDemoPlot{false};
  static bool s_showDemoIcons{false};
#endif
  if(ImGui::BeginMenu("File"))
  {
    if(ImGui::MenuItem(ICON_MS_FILE_OPEN " Open file", ""))
    {
      prmScene.sceneToLoadFilename = nvgui::windowOpenFileDialog(m_app->getWindowHandle(), "Load ply file",
                                                                 "All Files|*.ply;*.spz|PLY Files|*.ply|SPZ files|*.spz");
    }
    if(ImGui::MenuItem(ICON_MS_RESTORE_PAGE " Re Open", "F5", false, m_loadedSceneFilename != ""))
    {
      prmScene.sceneToLoadFilename = m_loadedSceneFilename;
    }
    if(ImGui::BeginMenu(ICON_MS_HISTORY " Recent Files"))
    {
      for(const auto& file : m_recentFiles)
      {
        if(ImGui::MenuItem(file.string().c_str()))
        {
          prmScene.sceneToLoadFilename = file;
        }
      }
      ImGui::EndMenu();
    }
    ImGui::Separator();
    if(ImGui::MenuItem(ICON_MS_FILE_OPEN " Open project", ""))
    {
      prmScene.projectToLoadFilename =
          nvgui::windowOpenFileDialog(m_app->getWindowHandle(), "Load project file", "VKGS Files|*.vkgs");
    }
    if(ImGui::BeginMenu(ICON_MS_HISTORY " Recent projects"))
    {
      for(const auto& file : m_recentProjects)
      {
        if(ImGui::MenuItem(file.string().c_str()))
        {
          prmScene.projectToLoadFilename = file;
        }
      }
      ImGui::EndMenu();
    }
    
    ImGui::Separator();
    if(ImGui::MenuItem(ICON_MS_SCAN_DELETE " Close", ""))
    {
      deinitAll();
    }
    ImGui::Separator();
    if(ImGui::MenuItem(ICON_MS_EXIT_TO_APP " Exit", "Ctrl+Q"))
    {
      close_app = true;
    }
    ImGui::EndMenu();
  }
  if(ImGui::BeginMenu("View"))
  {
    ImGui::MenuItem(ICON_MS_BOTTOM_PANEL_OPEN " V-Sync", "Ctrl+Shift+V", &v_sync);
    ImGui::MenuItem(ICON_MS_DATA_TABLE " Renderer Statistics", nullptr, &m_showRendererStatistics);
    ImGui::MenuItem(ICON_MS_DATA_TABLE " Memory Statistics", nullptr, &m_showMemoryStatistics);
#ifndef NDEBUG
    ImGui::MenuItem(ICON_MS_DATA_TABLE " Shader Debugging", nullptr, &m_showShaderDebugging);
#endif
    ImGui::EndMenu();
  }
#ifndef NDEBUG
  if(ImGui::BeginMenu("Debug"))
  {
    ImGui::MenuItem("Show ImGui Demo", nullptr, &s_showDemo);
    ImGui::MenuItem("Show ImPlot Demo", nullptr, &s_showDemoPlot);
    ImGui::MenuItem("Show Icons Demo", nullptr, &s_showDemoIcons);
    ImGui::EndMenu();
  }
#endif  // !NDEBUG

  // Shortcuts
  if(ImGui::IsKeyPressed(ImGuiKey_Space))
  {
    m_lastLoadedCamera = (m_lastLoadedCamera + 1) % m_cameraSet.size();
    m_cameraSet.loadPreset(m_lastLoadedCamera, false);
    m_requestUpdateShaders = true;
  }
  if(ImGui::IsKeyPressed(ImGuiKey_Q) && ImGui::IsKeyDown(ImGuiKey_LeftCtrl))
  {
    close_app = true;
  }

  if(ImGui::IsKeyPressed(ImGuiKey_V) && ImGui::IsKeyDown(ImGuiKey_LeftCtrl) && ImGui::IsKeyDown(ImGuiKey_LeftShift))
  {
    v_sync = !v_sync;
  }
  if(ImGui::IsKeyPressed(ImGuiKey_F5))
  {
    if(!m_recentFiles.empty())
      prmScene.sceneToLoadFilename = m_recentFiles[0];
  }
  if(ImGui::IsKeyPressed(ImGuiKey_F1))
  {
    std::string statsFrame;
    std::string statsSingle;
    // print old stats
    nvutils::Logger::getInstance().log(nvutils::Logger::eSTATS, "ParameterSequence %d \"%s\" = {\n%s\n%s}\n", 0,
                                       "F1 pressed ", statsFrame.c_str(), statsSingle.c_str());
  }
  if(ImGui::IsKeyPressed(ImGuiKey_1))
    prmSelectedPipeline = PIPELINE_VERT;
  if(ImGui::IsKeyPressed(ImGuiKey_2))
    prmSelectedPipeline = PIPELINE_MESH;

  // hot rebuild of shaders only if scene exist
  if(ImGui::IsKeyPressed(ImGuiKey_R))
  {
    if(!m_loadedSceneFilename.empty())
      m_requestUpdateShaders = true;
    else
      std::cout << "No scene loaded, cannot rebuild shader" << std::endl;
  }
  if(close_app)
  {
    m_app->close();
  }
#ifndef NDEBUG
  if(s_showDemo)
  {
    ImGui::ShowDemoWindow(&s_showDemo);
  }
  if(s_showDemoPlot)
  {
    //ImPlot::ShowDemoWindow(&s_showDemoPlot);
  }
  if(s_showDemoIcons)
  {
    //nvgui::showDemoIcons();
  }
#endif  // !NDEBUG

  if(m_app->isVsync() != v_sync)
  {
    m_app->setVsync(v_sync);
  }

}

void GaussianSplattingUI::onFileDrop(const std::filesystem::path& filename)
{
  // extension To lower case
  std::string extension = filename.extension().string();
  std::transform(extension.begin(), extension.end(), extension.begin(), ::tolower);

  //
  if(extension == ".ply")
    prmScene.sceneToLoadFilename = filename;
  else if(extension == ".spz")
    prmScene.sceneToLoadFilename = filename;
  else if(extension == ".vkgs")
    prmScene.projectToLoadFilename = filename;
  else if(extension == ".obj")
    prmScene.meshToImportFilename = filename;
  else
    std::cout << "Error: unsupported file extension " << extension << std::endl;
}

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
  // Handle project loading, may trigger a scene loading
  loadProjectIfNeeded();

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

std::filesystem::path makeAbsolutePath(const std::filesystem::path& base, const std::string& relativePath)
{
  return std::filesystem::absolute(base / relativePath);
}

// some macros to fetch jsoin values only if exist and affect to val

#define LOAD1(val, item, name)                                                                                         \
  if((item).contains(name))                                                                                            \
  (val) = (item)[name]

#define LOAD2(val, item, name)                                                                                         \
  if((item).contains(name))                                                                                            \
  (val) = {(item)[name][0], (item)[name][1]}

#define LOAD3(val, item, name)                                                                                         \
  if((item).contains(name))                                                                                            \
  (val) = {(item)[name][0], (item)[name][1], (item)[name][2]}

// This method is multi pass
bool GaussianSplattingUI::loadProjectIfNeeded()
{
  // Nothing to load
  if(prmScene.projectToLoadFilename.empty())
    return true;

  auto path = prmScene.projectToLoadFilename.string();

  // load the json and set loading status
  if(!loadingProject)
  {
    if(!m_loadedSceneFilename.empty())
      ImGui::OpenPopup("Load .vkg project file ?");

    // Always center this window when appearing
    ImVec2 center = ImGui::GetMainViewport()->GetCenter();
    ImGui::SetNextWindowPos(center, ImGuiCond_Appearing, ImVec2(0.5f, 0.5f));

    bool doReset = true;

    if(ImGui::BeginPopupModal("Load .vkg project file ?", NULL, ImGuiWindowFlags_AlwaysAutoResize))
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
      std::cout << "Opening project file " << path << std::endl;

      std::ifstream i(path);
      if(!i.is_open())
      {
        std::cout << "Error : unable to open project file " << path << std::endl;
        prmScene.projectToLoadFilename = "";
        return false;
      }

      try
      {
        i >> data;
      }
      catch(...)
      {
        std::cout << "Error : invalid project file " << path << std::endl;
        prmScene.projectToLoadFilename = "";
        return false;
      }
      i.close();

      loadingProject = true;

      // Initiate SplatSet loading
      if(!data["splats"].empty())
      {
        const auto& item             = data["splats"][0];
        prmScene.sceneToLoadFilename = makeAbsolutePath(std::filesystem::path(path).parent_path(), item["path"]);
      }
    }

    // Will do the rest of the work on next call when splatset is loaded
    return true;
  }

  // we skip until the splat set is being loaded
  if(m_plyLoader.getStatus() != PlyLoaderAsync::State::E_READY)
    return true;

  // we finalize
  loadingProject                 = false;
  prmScene.projectToLoadFilename = "";

  try
  {
    // Renderer
    if(data.contains("renderer"))
    {
      const auto& item = data["renderer"];

      if(item.contains("vsync"))
        m_app->setVsync(item["vsync"]);

      LOAD1(prmSelectedPipeline, item, "pipeline");

      LOAD1(prmRender.maxShDegree, item, "maxShDegree");
      LOAD1(prmRender.opacityGaussianDisabled, item, "opacityGaussianDisabled");
      LOAD1(prmRender.showShOnly, item, "showShOnly");
      LOAD1(prmRender.visualize, item, "visualize");
      LOAD1(prmRender.wireframe, item, "wireframe");

      LOAD1(prmRaster.cpuLazySort, item, "cpuLazySort");
      LOAD1(prmRaster.distShaderWorkgroupSize, item, "distShaderWorkgroupSize");
      LOAD1(prmRaster.fragmentBarycentric, item, "fragmentBarycentric");
      LOAD1(prmRaster.frustumCulling, item, "frustumCulling");
      LOAD1(prmRaster.meshShaderWorkgroupSize, item, "meshShaderWorkgroupSize");
      LOAD1(prmRaster.pointCloudModeEnabled, item, "pointCloudModeEnabled");
      LOAD1(prmRaster.sortingMethod, item, "sortingMethod");

      LOAD1(prmFrame.frameSampleMax, item, "temporalSamplesCount");

    }
    // Splat global options
    if(data.contains("splatsGlobals"))
    {
      const auto& item = data["splatsGlobals"];

      LOAD1(prmData.dataStorage, item, "dataStorage");
      LOAD1(prmData.shFormat, item, "shFormat");


      m_requestUpdateSplatData = true;
      m_requestUpdateSplatAs   = true;
    }
    // Parse splat settings
    if(data.contains("splats"))
    {
      if(!data["splats"].empty())
      {
        const auto& item = data["splats"][0];
        LOAD3(m_splatSetVk.translation, item, "position");
        LOAD3(m_splatSetVk.rotation, item, "rotation");
        LOAD3(m_splatSetVk.scale, item, "scale");

        computeTransform(m_splatSetVk.scale, m_splatSetVk.rotation, m_splatSetVk.translation, m_splatSetVk.transform,
                         m_splatSetVk.transformInverse);

        // delay update of Acceleration Structures if not using ray tracing
        m_requestDelayedUpdateSplatAs = true;
      }
    }

    // Load all the meshes
    if(data.contains("meshes"))
    {
      auto meshId = 0;
      for(const auto& item : data["meshes"])
      {
        std::string relPath;
        LOAD1(relPath, item, "path");
        if(relPath.empty())
          continue;

        auto meshPath = makeAbsolutePath(std::filesystem::path(path).parent_path(), relPath);
        if(!m_meshSetVk.loadModel(meshPath.string()))
        {
          meshId++;
          continue;
        }
        // Access to newly created mesh/instance
        auto& instance = m_meshSetVk.instances.back();
        auto& mesh     = m_meshSetVk.meshes[instance.objIndex];

        // Transform
        LOAD3(instance.translation, item, "position");
        LOAD3(instance.rotation, item, "rotation");
        LOAD3(instance.scale, item, "scale");
        computeTransform(instance.scale, instance.rotation, instance.translation, instance.transform, instance.transformInverse);

        // Materials
        if(item.contains("materials"))
        {
          auto matId = 0;
          for(const auto& matItem : item["materials"])
          {
            auto& mat = mesh.materials[matId];
            LOAD3(mat.ambient, matItem, "ambient");
            LOAD3(mat.diffuse, matItem, "diffuse");
            LOAD1(mat.illum, matItem, "illum");
            LOAD1(mat.ior, matItem, "ior");
            LOAD1(mat.shininess, matItem, "shininess");
            LOAD3(mat.specular, matItem, "specular");
            LOAD3(mat.transmittance, matItem, "transmittance");

            matId++;
          }
          m_meshSetVk.updateObjMaterialsBuffer(meshId);
        }

        meshId++;
      }
      m_requestUpdateMeshData = true;
      m_requestUpdateShaders  = true;
    }

    // Parse camera
    if(data.contains("camera"))
    {
      auto&  item = data["camera"];
      Camera cam;
      LOAD1(cam.model, item, "model");
      LOAD3(cam.ctr, item, "ctr");
      LOAD3(cam.eye, item, "eye");
      LOAD3(cam.up, item, "up");
      LOAD1(cam.fov, item, "fov");
      LOAD1(cam.dofEnabled, item, "dofEnabled");
      LOAD1(cam.focusDist, item, "focusDist");
      LOAD1(cam.aperture, item, "aperture");
      m_cameraSet.setCamera(cam);
    }
    // Parse camera presets
    if(data.contains("cameras"))
    {
      for(const auto& item : data["cameras"])
      {
        Camera cam;
        LOAD1(cam.model, item, "model");
        LOAD3(cam.ctr, item, "ctr");
        LOAD3(cam.eye, item, "eye");
        LOAD3(cam.up, item, "up");
        LOAD1(cam.fov, item, "fov");
        LOAD1(cam.dofEnabled, item, "dofEnabled");
        LOAD1(cam.focusDist, item, "focusDist");
        LOAD1(cam.aperture, item, "aperture");
        m_cameraSet.createPreset(cam);
      }
    }
    // Parse lights
    if(data.contains("lights"))
    {
      bool defaultLight = true;
      for(const auto& item : data["lights"])
      {
        // A default light already exists, we only modify it
        uint64_t id = 0;
        if(!defaultLight)
        {
          id = m_lightSet.createLight();
        }
        auto& light = m_lightSet.getLight(id);
        LOAD1(light.type, item, "type");
        LOAD3(light.position, item, "position");
        LOAD1(light.intensity, item, "intensity");
        defaultLight = false;
      }
      m_requestUpdateLightsBuffer = true;
    }

    return true;
  }
  catch(...)
  {
    return false;
  }
}

}  // namespace vk_gaussian_splatting
