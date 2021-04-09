# Adapted from the setup script in the g2opy repository

source ./venv/bin/activate

if [ ! -d "./g2opy/" ]
then
  git clone https://github.com/uoip/g2opy.git
fi

cd g2opy || exit

# Fix the setup file
cp ../working_g2opy_build_assets/new_setup.py setup.py

# Fix the eigen file
cp ../working_g2opy_build_assets/fixed_eigen_types.h python/core/eigen_types.h

if [ ! -d "./build/" ]
then
  mkdir build
fi

cd build || exit
cmake ..
make -j12
cd ..

python setup.py install