# Copyright 2021 The Pigweed Authors
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
"""Utilities for fuzzing."""

load("@rules_fuzzing//fuzzing:cc_defs.bzl", "cc_fuzz_test")

def pw_cc_fuzz_test(**kwargs):
    """Wrapper for cc_fuzz_test that adds required Pigweed dependencies.

    Args:
        **kwargs: Arguments to be augmented.
    """
    kwargs["testonly"] = True
    kwargs["deps"].append("//pw_fuzzer:libfuzzer")
    kwargs["deps"].append(str(Label("//pw_build:default_link_extra_lib")))

    # Pigweed's fuzztest headers needs deduction guides.
    if "features" not in kwargs:
        kwargs["features"] = []
    kwargs["features"].append("-ctad_warnings")

    # TODO: b/292628774 - Only linux is supported for now.
    if "target_compatible_with" not in kwargs:
        kwargs["target_compatible_with"] = ["@platforms//os:linux"]
    cc_fuzz_test(**kwargs)
