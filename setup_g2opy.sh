#!/bin/bash
# Installs g2opy. Adapted from the setup script in the g2opy repository. It may be run from any directory.

VENV_DIR_REL_PATH="./venv"
VENV_ACTIVATE_PATH="${VENV_DIR_REL_PATH}/bin/activate"

# To handle this script being run from any directory, cd into the repository's root directory
SCRIPT_PATH=$(relpath "$0")
REPODIR=$(dirname "$(dirname "$SCRIPT_PATH")")
cd "$REPODIR" || exit

if [ ! -d ${VENV_DIR_REL_PATH} ]
then
  echo "Could not find a directory at ${VENV_DIR_REL_PATH} to be used as the venv directory; will proceed with Python at: $(which python3)"
else
  echo "Activating the virtual environment with activate script at ${VENV_ACTIVATE_PATH}"
  # shellcheck disable=SC1090
  source "${VENV_ACTIVATE_PATH}" || exit
  which python3
fi


if [ ! -d "./g2opy/" ]  # Ignore cloning the repository if it already exists
then
  git clone https://github.com/occamLab/g2opy || exit
fi
cd g2opy || exit

if [ ! -d "./build/" ]  # Make a new build directory if one doesn't already exist
then
  mkdir build || exit
fi
cd build || exit

cmake .. || exit
make "-j$(nproc --all)" || exit
cd ..

echo "Using python version: $(python3 --version)"
python3 setup.py install || exit
cd "${REPODIR}" || exit

