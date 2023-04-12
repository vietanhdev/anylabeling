# Most of the steps were done following this: https://pyimagesearch.com/2020/02/03/how-to-use-opencvs-dnn-module-with-nvidia-gpus-cuda-and-cudnn/

# Download openCV and create your environment:
cd ~
mkdir .opencv_drivers
cd .opencv_drivers/
wget -O opencv.zip https://github.com/opencv/opencv/archive/4.6.0.zip
wget -O opencv_contrib.zip https://github.com/opencv/opencv_contrib/archive/4.6.0.zip
unzip opencv.zip
unzip opencv_contrib.zip
mv opencv-4.6.0 opencv
mv opencv_contrib-4.6.0 opencv_contrib
mkvirtualenv opencv_cuda
pip install numpy
cd opencv
mkdir build
cd build

# IMPORTANT!: CUDA_ARCH_BIN version must be seen here: https://developer.nvidia.com/cuda-gpus
# I got RTX 2060, so in my case CUDA_ARCH_BIN=7.5
# gcc version > 8 are not supported so set CMAKE_C_COMPILER=/usr/bin/gcc-8

cmake -D CMAKE_BUILD_TYPE=RELEASE \
	-D CMAKE_INSTALL_PREFIX=/usr/local \
	-D INSTALL_PYTHON_EXAMPLES=ON \
	-D INSTALL_C_EXAMPLES=OFF \
	-D OPENCV_ENABLE_NONFREE=ON \
	-D WITH_CUDA=ON \
	-D WITH_CUDNN=ON \
	-D OPENCV_DNN_CUDA=ON \
	-D ENABLE_FAST_MATH=1 \
	-D CUDA_FAST_MATH=1 \
	-D CUDA_ARCH_BIN=7.5 \
	-D WITH_CUBLAS=1 \
	-D OPENCV_EXTRA_MODULES_PATH=~/.opencv_drivers/opencv_contrib/modules \
	-D HAVE_opencv_python3=ON \		
	-D PYTHON_EXECUTABLE=~/.virtualenvs/opencv_cuda/bin/python \
	-D BUILD_EXAMPLES=ON \
	-D CMAKE_C_COMPILER=/usr/bin/gcc-8 ..

# Compile your openCV built before
make -j12
sudo make install
sudo ldconfig


# ln -s command is used to create a symbolic link, also known as a symlink or soft link
cd ~/.virtualenvs/opencv_cuda/lib/python3.8/site-packages/
ln -s /usr/local/lib/python3.8/site-packages/cv2/python-3.8/cv2.cpython-38-x86_64-linux-gnu.so cv2.so

# Install all requirements for anylabeling
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Uninstall openCV dependencies, since dnn module installed from source is not recognize by pip
pip uninstall opencv-contrib-python-headless opencv-python-headless










