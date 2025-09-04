# Copyright 2024 The Pigweed Authors
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.
"""Generates a module map for the include directories."""

load("@bazel_skylib//lib:paths.bzl", "paths")
load("@bazel_skylib//rules/directory:providers.bzl", "DirectoryInfo")

def _impl(ctx):
    module_map_file = ctx.actions.declare_file("module.modulemap")

    header_paths = list()
    files = list()
    for directory in ctx.attr.include_directories:
        header_paths.append(directory[DirectoryInfo].path)
        files.append(directory[DirectoryInfo].transitive_files)

    # The paths in the module map must be relative to the path to the module map
    # itself, but the header_paths provided as arguments must be relative to
    # the execroot. This prefix is just enough ../ to transform between the two.
    prefix = "../" * len(paths.dirname(module_map_file.path).split("/"))

    ctx.actions.run(
        inputs = depset([], transitive = files),
        executable = ctx.executable._module_map_generator,
        arguments = [module_map_file.path, prefix] + header_paths,
        outputs = [module_map_file],
    )

    return [DefaultInfo(files = depset([module_map_file]))]

builtin_module_map = rule(
    implementation = _impl,
    attrs = {
        "include_directories": attr.label_list(
            providers = [DirectoryInfo],
            doc = """Directories in which to search for builtin headers.""",
        ),
        "_module_map_generator": attr.label(
            default = Label("//pw_toolchain/cc:generate_module_map"),
            executable = True,
            cfg = "exec",
        ),
    },
    provides = [DefaultInfo],
    doc = """Generates a module map for the include directories.""",
)
