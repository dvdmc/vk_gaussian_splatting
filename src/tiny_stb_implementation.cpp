/*
 * Copyright (c) 2024-2026, NVIDIA CORPORATION.  All rights reserved.
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
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

// tinygltf and stb_image are header-only libraries: each header contains both
// declarations and definitions, guarded by an implementation macro.  The macro
// must be defined in exactly one translation unit — if no file defines it you
// get linker errors; if multiple files define it you get duplicate symbols.
// This file is that one translation unit.
//
// TINYGLTF_NO_EXTERNAL_IMAGE suppresses tinygltf's built-in image loading so
// that nvvkgltf::SceneVk can handle texture upload to the GPU directly.
//
// The warning pragmas silence signed/unsigned and size_t truncation warnings
// in third-party code we do not control.

#define TINYGLTF_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#define TINYGLTF_NO_EXTERNAL_IMAGE
#pragma warning(push)
#pragma warning(disable : 4018)  // signed/unsigned mismatch
#pragma warning(disable : 4267)  // conversion from 'size_t' to 'uint32_t', possible loss of data
#pragma warning(disable : 4996)
#include <tinygltf/tiny_gltf.h>
#pragma warning(pop)
