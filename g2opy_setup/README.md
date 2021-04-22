# Summary

Run `install_g2opy.sh` from any directory. At the end of the script, g2opy's Python setup script is called with the default python interpreter (make sure to source your virtual environment before calling this script to get the desired Python version); it was found to work with Python 3.8.5 on Ubuntu 20.0.4.

# Description

A working build of g2opy on Ubuntu 20.0.4 was achieved using Python 3.8.5. Problems were encountered with Python 3.6.x but were resolved with Python 3.8.5 (although the root cause for this is currently unknown). A virtual environment was used, and only the dependencies mentioned in `requirements.txt` were installed.

A script to install g2opy is in this directory: `install_g2opy.sh`. It clones the g2opy repository into the root directory and follows the installation process outlined in the [installation instructions contained in the g2opy README](https://github.com/uoip/g2opy#Installation), but with one important change: the script replaces a file in the g2opy source code, `g2opy/python/core/eigen_types.h`, with `<this_directory>/fixed_eigen_types.h`. As noted in this repository's README, there is a [pull request]((https://github.com/uoip/g2opy/pull/16)) to the g2opy repository that contains a fix for breaking builds. As the pull request is still open, `fixed_eigen_types.h` contains the fixes specified in the commit.

### Code changes in `fixed_eigen_types.h`

*Changes made to lines 185-189.*

Original:

```c
.def("x", (double (Eigen::Quaterniond::*) () const) &Eigen::Quaterniond::x)
.def("y", (double (Eigen::Quaterniond::*) () const) &Eigen::Quaterniond::y)
.def("z", (double (Eigen::Quaterniond::*) () const) &Eigen::Quaterniond::z)
.def("w", (double (Eigen::Quaterniond::*) () const) &Eigen::Quaterniond::w)
```

New:

```c
.def("x", (const double&  (Eigen::Quaterniond::*) () const) &Eigen::Quaterniond::x)
.def("y", (const double&  (Eigen::Quaterniond::*) () const) &Eigen::Quaterniond::y)
.def("z", (const double&  (Eigen::Quaterniond::*) () const) &Eigen::Quaterniond::z)
.def("w", (const double&  (Eigen::Quaterniond::*) () const) &Eigen::Quaterniond::w)
```
