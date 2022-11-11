# wavefilter
<!-- ALL-CONTRIBUTORS-BADGE:START - Do not remove or modify this section -->
[![All Contributors](https://img.shields.io/badge/all_contributors-1-orange.svg?style=flat-square)](#contributors-)
<!-- ALL-CONTRIBUTORS-BADGE:END -->

[![PyPI - Version](https://img.shields.io/pypi/v/wavefilter.svg)](https://pypi.org/project/wavefilter)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/wavefilter.svg)](https://pypi.org/project/wavefilter)
[![CI](https://github.com/benkrikler/wavefilter/actions/workflows/ci.yml/badge.svg)](https://github.com/benkrikler/wavefilter/actions/workflows/ci.yml)
<!-- ALL-CONTRIBUTORS-BADGE:START - Do not remove or modify this section -->
[![All Contributors](https://img.shields.io/badge/all_contributors-13-orange.svg?style=flat-square)](#contributors)
<!-- ALL-CONTRIBUTORS-BADGE:END -->

-----

**Table of Contents**

- [Overview](#overview)
- [Installation](#installation)
- [Developing](#developing)
- [License](#license)

## Overview
*TODO: Give a brief explanation here*

See the [design choices](docs/design.md) document for more details.

## Installation

```console
pip install wavefilter
```

## Developing

### Pre-commit tests
Static code tests and formatting checks can be run using [pre-commit](pre-commit.com). Install pre-commit, then initialise the hooks for this project with:
```
pre-commit install
```

When you next commit a change, pre-commit will run various formatting and static code checks, in some cases fixing any issues automatically (see e.g. `black` and `isort`).

To manually trigger these checks on all files, you can do something like:

```console
pre-commit run --all-files
```
See pre-commit's built-in help for more guidance.

### Unit testing
Run the built-in unit tests with `pytest`:

```
python -m pytest
```

## Contributors

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table>
  <tbody>
    <tr>
      <td align="center"><a href="https://jakebolam.com"><img src="https://avatars.githubusercontent.com/u/3534236?v=4?s=100" width="100px;" alt="Jake Bolam"/><br /><sub><b>Jake Bolam</b></sub></a><br /><a href="#ideas-jakebolam" title="Ideas, Planning, & Feedback">🤔</a> <a href="#design-jakebolam" title="Design">🎨</a> <a href="https://github.com/benkrikler/wavefilter/commits?author=jakebolam" title="Documentation">📖</a> <a href="https://github.com/benkrikler/wavefilter/commits?author=jakebolam" title="Tests">⚠️</a> <a href="https://github.com/benkrikler/wavefilter/commits?author=jakebolam" title="Code">💻</a></td>
    </tr>
  </tbody>
</table>

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->

See [CONTRIBUTING.md](./CONTRIBUTING.md) for how to contribute.

## License

`wavefilter` is distributed under the terms of the [MIT](https://spdx.org/licenses/MIT.html) license.


*Based on python package template from https://github.com/benkrikler/my_python_cookie*
