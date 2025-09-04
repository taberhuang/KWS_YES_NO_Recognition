# Copyright 2025 The Pigweed Authors
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
"""Defines the pw_size_report rule, used to generate a size report table."""

load("//pw_bloat/private:pw_bloat_report.bzl", _pw_bloat_report = "pw_bloat_report")

def pw_size_report(**kwargs):
    _pw_bloat_report(base = None, **kwargs)
