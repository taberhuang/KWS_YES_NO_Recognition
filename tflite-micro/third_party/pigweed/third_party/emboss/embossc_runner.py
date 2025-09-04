# Copyright 2022 The Pigweed Authors
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
"""Embossc wrapper used by the GN templates."""

import sys
import importlib.machinery
import importlib.util

loader = importlib.machinery.SourceFileLoader("embossc", sys.argv[1])
spec = importlib.util.spec_from_loader(loader.name, loader)
assert spec is not None
main_module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(main_module)

sys.exit(main_module.main(sys.argv[1:]))
