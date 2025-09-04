// Copyright 2023 The Pigweed Authors
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
#![cfg_attr(feature = "nightly", feature(type_alias_impl_trait))]

#[cfg(test)]
mod tests {
    use pw_format_example_macro::example_macro;

    #[test]
    fn test() {
        let string = example_macro!("the answer: ", "%d", 42);
        assert_eq!(&string, "the answer: 42");
    }
}
