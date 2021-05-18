# Installs g2opy. Adapted from the setup script in the g2opy repository. It may be run from any directory.

# To handle this script being run from any directory, cd into the repository's root directory
# Absolute path to this script, e.g. /home/user/bin/foo.sh
SCRIPT=$(readlink -f "$0")
# Get the repository directory
REPODIR=$(dirname "$(dirname "$SCRIPT")")
echo "Changing directory into ${REPODIR}"
cd "$REPODIR" || exit

# Ignore cloning the repository if it already exists
if [ ! -d "./g2opy/" ]
then
  git clone https://github.com/uoip/g2opy.git
fi

cd g2opy || exit

# Replace the eigen file
echo "Replacing the eigen_types.h file with the corrected one"
cp ../g2opy_setup/fixed_eigen_types.h python/core/eigen_types.h || exit

# Make a new build directory if one doesn't already exist
if [ ! -d "./build/" ]
then
  mkdir build
fi

cd build || exit
cmake ..
make -j12 || exit
cd ..

echo "Using python version:"
python --version
echo "Running: python g2opy/setup.py install"
python setup.py install
