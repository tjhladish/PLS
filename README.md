# Partial Least Squares (PLS) regression in C++, using Eigen.

## Quick Start

From a generic overall projects root folder:

```
git clone git@github.com:tjhladish/PLS.git         # get the repository
```

As a submodule within another project, navigate to where you'll be collecting libraries (e.g. `lib`, `libs`, `external`), then:

```
git submodule add git@github.com:tjhladish/PLS.git # get the repository
```

Then

```
cd PLS                                     # enter newly created project / submodule directory
git submodule update --init --recursive    # pull down the associated submodules (Eigen)
sudo make install                          # setup build folder
PLS ../toyX.csv ../toyY.csv 2              # test executable
```

## Contributing

See our [contributing guidelines](CONTRIBUTING.md).

## Copyright

Authors: Thomas Hladish, Eugene Melamud, Carl A. B. Pearson  
Copyright 2012, 2013, 2014 Thomas Hladish, Eugene Melamud  
Copyright 2023 Thomas Hladish, Eugene Melamud, Carl A. B. Pearson

## License

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see <http://www.gnu.org/licenses/>.
