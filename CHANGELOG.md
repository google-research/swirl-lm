# Changelog

<!--

Changelog follow the https://keepachangelog.com/ standard (at least the headers)

This allow to:

* auto-parsing release notes during the automated releases from github-action:
  https://github.com/marketplace/actions/pypi-github-auto-release
* Have clickable headers in the rendered markdown

To release a new version (e.g. from `1.0.0` -> `2.0.0`):

* Create a new `# [2.0.0] - YYYY-MM-DD` header and add the current
  `[Unreleased]` notes.
* At the end of the file:
  * Define the new link url:
  `[2.0.0]: https://github.com/google-research/swirl_lm/compare/v1.0.0...v2.0.0`
  * Update the `[Unreleased]` url: `v1.0.0...HEAD` -> `v2.0.0...HEAD`

## [Unreleased]
* Fixing pyproject.toml: 1. Correcting the dependent package absl to absl-py.
 2. Adding back optional-dependencies.
## [0.0.0] - 2022-08-15

* Initial release

[Unreleased]: https://github.com/google-research/swirl_lm/compare/v0.0.0...HEAD
[0.0.0]: https://github.com/google-research/swirl_lm/releases/tag/v0.0.0

-->
