# Introduction

To get the build of g2opy working on my (Duncan's) system (Ubuntu 20.0.4 using Python 3.8.5 in a virtual environment), I had to make the following two changes to source code in the g2opy repository. I did this by making a setup script, `g2opy_setup.sh` for g2opy that is essentially the same as the [installation instructions contained in the g2opy README](https://github.com/uoip/g2opy#Installation), but with one important change: the script replaces two files in the g2opy source code with the files in this directory.

# Change 1

As noted in this repository's README, there is a [pull request]((https://github.com/uoip/g2opy/pull/16)) to the g2opy repository that contains a fix for breaking builds. As the pull request is still open, I copied the changed file from the pull request into `fixed_eigen_types.h`.

## Code changes

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

# Change 2

I encountered an assertion error (from line 21 of the [setup.py](https://github.com/uoip/g2opy/blob/master/setup.py)) when the script attempted the `python setup.py install` command, and found a fix in [this pull request](https://github.com/uoip/pangolin/issues/20). Although the error associated with the setup scripted changed to `error: invalid command 'installll'` after change no. 1 was applied, I found that the fix still works and implemented the changes in `new_setup.py`.

## Code changes

Original:

```python
install_dirs = [install_dir]
lib_file = glob.glob(__library_file__)
assert len(lib_file) == 1 and len(install_dirs) >= 1
```

New:

```python
install_dir = get_python_lib()
lib_file = glob.glob(__library_file__)
assert len(lib_file) == 1
```
