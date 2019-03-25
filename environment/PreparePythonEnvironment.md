# Prepare Python Environment

To set up a reliable python environment on your system for machine learning is not a trivial task.
During the devlopment, implementation and evaluation of machine learning algorithms, we rely on various Python packages for managing the data, accelerating the computation, visualizing the results and debugging in case error occurs.
In this tutuorial, I provide the steps that I used to set up my machine learning tool chain, and solutions to some of the issues I have encountered during the process.

## MiniConda3

**Conda** is a powerful language-agnostic package manager with cross-platform support.
I use **[MiniConda3](https://docs.conda.io/en/latest/miniconda.html)**, which is a slimmed-down version of Anaconda, as a clean start point for setting up my Python environment.

If you have previous experience with Python programming, you are probably aware of other tools such as `pip`, `easy_install` or `virtualenv`.
Both `pip` and `easy_install` can manage and setup Python packages.
However, in ML development and implementation, we also need packages with non-Python library dependencies, such as *HDF5* for large dataset management, Intel *MKL* for improved `numpy` computation performance, and `vtk` for 3D visualization with `mayavi`.

Moreover, as **Conda** has support for all operating systems including Windows 10, Linux and Mac OS, if you have multiple computers for ML development, **Conda** can make sure that you have the same version of packages installed among all computers.

The other important benefits of using **Conda** to manage the packages is the easy setup of `virtualenv` support.
Virtual environment support allows you to have multiple python environments co-exist on your system for different projects.
If one virtual environment crashes during a failed upgrade or installation of pre-released package, it won't affect other projects using different python environment.

Nevertheless, there are still cases where the Python packages we want to install is not yet available in Conda package manager.
In such cases, `pip` can still be used to install the Python package.
The packages installed with `pip` will serve as an overlay on top of the Conda packages.
As a result, precautions are needed so that you do not accidentally install a package that Conda has already installed, as the dependencies management is not shared between `pip` and Conda, and thus it may cause various troubles down the road or destroy your environment.

### Install MiniConda3

To install **[MiniConda3](https://docs.conda.io/en/latest/miniconda.html)** for your operating system, simply go to the [download page of MiniConda3](https://docs.conda.io/en/latest/miniconda.html) and follow the instruction for your operating system.
In this tutorial, I assume that **MiniConda3** is installed to `C:\MiniConda3` on Windows, or `/Users/<username>/miniconda3` on Mac OS or Linux.

### Create Virtual Environment

Open conda prompt and create a virtual environment for your future ML project.
At the time of writing, the most recent python release is `3.7.1`.

```bash
(base) > conda create -n ml2019 python=3.7.1
```

Activate your new environment `ml2019` as follows:
```bash
(base) > conda activate ml2019
```

When virtual environment `ml2019` is activated, you can see `(ml2019)` before the prompt in the command line. You may also check the path of `python` and `pip` to see if the executables under the `ml2019` environment is used.

In MacOS/Linux:

```bash
(ml2019) > which python
/Users/tinghuiwang/miniconda3/envs/ml2019/bin/python

(ml2019) > which pip
/Users/tinghuiwang/miniconda3/envs/ml2019/bin/pip
```

In Windows:
```cmd
(ml2019) > where python
C:\Miniconda3\envs\ml2019\python.exe

(ml2019) > where pip
C:\Miniconda3\envs\ml2019\Scripts\pip.exe
```

### Install Basic Packages

Here are some commonly used packages that you may want to install:
- `numpy`: A fundamental package for scientific computing.
- `scipy`: A expanding set of scientific computing libraries.
- `scikit-learn`: Simple and efficient tools for data mining and analysis.
- `matplotlib`: 2D plotting library which produces publication quality figures.
- `h5py`: Pythonic interface to the HDF5 binary data format.
- `jupyter`: Jupyter notebook.

You can install them using the commands below.
Please make sure `ml2019` envrionment is activated.

```
(ml2019) > conda install numpy scipy scikit-learn matplotlib h5py jupyter
```

#### Fixes for Jupyter Notebook 5.7.6 MIME mismatch error

When starting Jupyter Notebook using command 
```cmd
(ml2019) > jupyter notebook
```

In the browser, under **Console** of **Developer Tools**, you may see the following error:

```
Refused to execute script from '<URL>' because its MIME type ('text/plain') is not executable, and strict MIME type checking is enabled.

Refused to execute script from 'http://localhost:8888/static/components/es6-promise/promise.min.js?v=f004a16cb856e0ff11781d01ec5ca8fe' because its MIME type ('text/plain') is not executable, and strict MIME type checking is enabled.
tree:1 Refused to execute script from 'http://localhost:8888/static/components/preact/index.js?v=00a2fac73c670ce39ac53d26640eb542' because its MIME type ('text/plain') is not executable, and strict MIME type checking is enabled.
tree:1 Refused to execute script from 'http://localhost:8888/static/components/proptypes/index.js?v=c40890eb04df9811fcc4d47e53a29604' because its MIME type ('text/plain') is not executable, and strict MIME type checking is enabled.
tree:1 Refused to execute script from 'http://localhost:8888/static/components/preact-compat/index.js?v=aea8f6660e54b18ace8d84a9b9654c1c' because its MIME type ('text/plain') is not executable, and strict MIME type checking is enabled.
tree:1 Refused to execute script from 'http://localhost:8888/static/components/requirejs/require.js?v=951f856e81496aaeec2e71a1c2c0d51f' because its MIME type ('text/plain') is not executable, and strict MIME type checking is enabled.
tree:1 Refused to execute script from 'http://localhost:8888/static/tree/js/main.min.js?v=ab9f3a6cf8347df927864d58cfad7931' because its MIME type ('text/plain') is not executable, and strict MIME type checking is enabled.
```

The above errors are caused by the **strict MIME checking** feature enabled starting with Jupyter notebook 5.7.6.
As the Tornado server hosts javascript files with `Content-Type: text/plain`, the browser reject loading the javascript.
The problem appears to manifest only on Windows hosted notebooks. 
The problem can partially be solved according to the [Jupyter Notebook Pull Request #4468](https://github.com/jupyter/notebook/pull/4468).
The `notebookapp.py` file can be located at `C:\Miniconda3\envs\ml2019\Lib\site-packages\notebook`.

## Tensorflow 2.0.0

Tensorflow is an end-to-end open source platform for machine learning.
It is now the most popular software framework for deep learning.

Tensorflow 2.0 is an important milestone for the framework focusing on exciting features such as eager execution, auto graph and simplified APIs.
A preview version of Tensorflow 2.0.0 is released on Mar. 5, 2019.
Please refer to the [Get started with TensorFlow 2.0](https://www.tensorflow.org/alpha) for installation details.

If you have an NVidia Graphic card with CUDA support, you can install CUDA 10.0 with CUDNN 7.4.x according to the [GPU Guide](https://www.tensorflow.org/install/gpu).
Note that the TensorFlow pre-compiled `pip` package is linked to specific versions of CUDA and CUDNN, always refer to the [GPU Guide](https://www.tensorflow.org/install/gpu) for the exact version of CUDA and CUDNN library to install.

## Mayavi 
`mayavi` enables 3D scientific data visualization and plotting in Python.

`mayavi` depends on `vtk` library for graphic rendering.
Here, we use **Conda** to manage `vtk` library.

```
(ml2019) > conda install vtk
```

However, **Conda** does not yet include a `mayavi` package for Python 3.7.1, we use `pip` to install `mayavi` library.

```
(ml2019) > pip install mayavi
```

## Test your environment

Please refer to the Python notebook `environment_test.ipynb` to test your environment installation.
