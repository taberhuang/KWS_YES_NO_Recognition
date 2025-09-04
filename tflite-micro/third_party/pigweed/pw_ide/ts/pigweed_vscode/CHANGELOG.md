# Change Log

## [1.9.5] - 2025-07-07

### Features

 - Revamped code intelligence integration using clangd.
 - Automatic target recognition for Bazel workflows.
 - A new python based commands generator (this is experimental) that is more
   accurate and simpler to maintain moving forward.

### Bug Fixes

 - Fix fish shell support for the commands generator ([b/428243657](https://issuetracker.google.com/428243657))

## [1.3.3]

- The `pigweed.activateBazeliskInNewTerminals` option has been set to a default
  of `false` because, as currently implemented, this feature interferes with
  interactive tasks like flashing scripts and `pw_console`.

## [1.3.2]

- When you manually run `Pigweed: Refresh Compile Commands`, we now show a
  progress notification so you know what's happening.

- Synchronizing settings between the committed settings (`settings.shared.json`)
  and the local settings (`settings.json`) is now dramatically faster (pretty
  much instantaneous).

- If you manually change the code analysis target by editing a settings file
  directly, the effect will now be the same as if you had run the
  `Pigweed: Select Code Analysis Target`.

## [1.3.1]

- Newly-launched integrated terminals will now automatically have the
  configured path to `bazelisk` patched into the shell path. This is essentially
  running `Pigweed: Activate Bazelisk in Terminal` automatically when launching
  a new integrated terminal. This behavior can be disabled by setting
  `pigweed.activateBazeliskInNewTerminals` to `false` in your settings.

## [1.3.0]

- You can now disable `clangd` code intelligence for files that are not built
  as part of the target you're using for code intelligence. Why would you want
  to do this? By default, `clangd` will try to *infer* compile commands for
  files that won't actually be built, producing incorrect and misleading code
  intelligence information ([see this doc](https://pigweed.dev/pw_ide/guide/vscode/code_intelligence.html#inactive-and-orphaned-source-files)
  for more information).

- File indicators in the file explorer and file tabs now show each file's status
  with regard to the currently-selected code intelligence target. For example,
  files that are not part of the target's build are faded out so you're not
  surprised when they don't show code intelligence ([learn more here](https://pigweed.dev/pw_ide/guide/vscode/code_intelligence.html#inactive-and-orphaned-source-files)).

- We now provide a mechanism for shared settings that can be committed to source
  control, along with personal user settings that also include ephemeral editor
  state, like the currently-selected code intelligence target and `clangd`
  settings. [Check out this doc](https://pigweed.dev/pw_ide/guide/vscode/#project-settings)
  for more details.

- Output from the compile commands refresh process is now streamed in real-time
  to the Pigweed output window.

- The method of finding the path to `clangd` in the toolchain brought in by
  Bazel is now more robust and reliable.

- Bundled tools (`bazelisk` and `buildifier`) now have their paths automatically
  updated whenever a new version of the extension is installed.

## [1.1.4]

- Properly supports [bzlmod](https://docs.bazel.build/versions/5.1.0/bzlmod.html)
  projects.

## [1.1.3]

- Fixes a project root inference bug.

## [1.1.2]

- Output from the compile commands refresh process is now streamed to the
  Pigweed output panel in real time, allowing you to monitor progress in the
  same way you would if you had invoked it from the terminal. Before, the output
  was just buffered and sent to the output window when complete.

- The extension will no longer attempt to infer the project root if one is set
  manually via [pigweed.projectRoot](https://pigweed.dev/pw_ide/guide/vscode/#pigweed.projectRoot).
  This prevents spurious failures if you have an uncommon project structure but
  have pointed the Pigweed extension in the right direction by specifying the
  project root.

- Missing command implementation stubs for bootstrap-based projects are no
  longer missing.

## [1.1.1]

- Adds support for [fish shell](https://fishshell.com/) to the
  `Pigweed: Activate Bazelisk in Terminal` command (in addition to the existing
  support for `bash` and `zsh`).

## [1.1.0]

- Adds `Pigweed: Activate Bazelisk in Terminal`: This command will modify the
  `$PATH` in your active integrated terminal to include the configured path to
  Bazelisk. This allows you to run Bazel actions via editor commands or via
  `bazelisk ...` invocations in the terminal, while working in the same Bazel
  environment.

## [1.0.0]

- Initial release of the new and improved Pigweed extension.
