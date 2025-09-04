// Copyright 2025 The Pigweed Authors
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not
// use this file except in compliance with the License. You may obtain a copy of
// the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
// WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
// License for the specific language governing permissions and limitations under
// the License.

import { settings, workingDir } from './settings/vscode';
import { fileTypeExists } from './utils';

/** Are CMake build files present in the workspace? */
export async function cmakeFilesArePresent(): Promise<boolean> {
  return fileTypeExists(`${workingDir.get()}/**/CMakeLists.txt`);
}

/** Should we try to generate compile commands for CMake targets? */
export async function shouldSupportCmake(): Promise<boolean> {
  const settingValue = settings.supportCmakeTargets();

  if (settingValue !== undefined && settingValue !== 'auto') {
    return settingValue;
  }

  return await cmakeFilesArePresent();
}
