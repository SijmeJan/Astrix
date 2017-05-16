Installation
=========================

Astrix is completely open-source and freely available from GitHub. It runs on both Mac and Linux machines (Windows is not supported) as long as CUDA is installed.

Installing CUDA
-------------------------------

Note that while Astrix does not require a GPU to run, it needs the CUDA compiler in order to be built. CUDA can be obtained for free from https://developer.nvidia.com/cuda-downloads. There are three parts that can be installed:

* CUDA toolkit
* CUDA samples
* CUDA driver

Only the CUDA toolkit is required for building Astrix. Installing the driver is only necessary on machines that you plan to use the GPU of (requires root access); installing the samples is optional but they include some useful tests to see if everything is working correctly. Unfortunately for Mac users, CUDA only works with specific versions of Xcode, see http://docs.nvidia.com/cuda/cuda-installation-guide-mac-os-x/index.html#prerequisites. For supported versions of Linux see http://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#system-requirements.

Installing Astrix
-------------------------------

Once CUDA has been installed correctly, it is time to download Astrix. At a terminal, type::

  git clone http://github.com/SijmeJan/Astrix

This will create a directory ``Astrix``. Now do::

  cd Astrix

Astrix needs to know where CUDA was installed. The easiest way is to make sure that the CUDA ``bin`` directory (for example ``/usr/local/cuda/bin``) is in your ``PATH`` environmental variable.  You can check if this is the case by typing ``which nvcc``. If this gives a result, you are good to go. If not, add the CUDA ``bin`` directory to your path, and then type::

  make astrix

Sit back and relax because this may take a while. When finished, Astrix is ready to use. The executable will be located in ``Astrix/bin``.

Advanced options
++++++++++++++++++++++++++

The compilation process can be sped up is you know what GPU you are going to use. If you are planning to run on a Tesla K20m, which has CUDA compute capability 3.5, issue::

  make astrix CUDA_COMPUTE=35

Not only will this speed up compilation, it will also speed up the code itself since the compiler can use a few tricks not available on devices with compute capability < 3.5. The minimum compute capability currently supported by nVidia is 3.0; while Astrix should work on devices with compute capability 2.x this is not supported by the standard build.

It is possible to override the standard CUDA installation and build with a specific version located in ``path/to/cuda``::

  make astrix CUDA_INSTALL_PATH=path/to/cuda

This can be useful when testing a new CUDA version.

By default, Astrix will perform floating point computations in single precision. It is possible to force double presicion through::

  make astrix ASTRIX_DOUBLE=1

Note that switching between double and single precision requires a complete rebuild: therefore first remove a previous build by entering::

  make clean

A simple visualisation program is included and can be built by::

  make visAstrix

This requires the OpenGL and glut libraries to be installed.

If you are interested in building this documentation locally, it can be built through::

  make doc

In order to clean up everything, enter::

  make allclean

which removes the Astrix build, any visAstrix build and all documentation.
