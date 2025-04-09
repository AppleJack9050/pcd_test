from setuptools import setup, Extension
import numpy as np

# Adding OpenCV to project
# ​************************

# Adding sources of the project
# ​*****************************

SOURCES = [
    "../cpp_utils/cloud/cloud.cpp",
    "grid_subsampling/grid_subsampling.cpp",
    "wrapper.cpp"
]

module = Extension(
    name="grid_subsampling",
    sources=SOURCES,
    include_dirs=[np.get_include()],  # 更直接的 NumPy 头文件获取方式
    extra_compile_args=[
        '-std=c++11',
        '-D_GLIBCXX_USE_CXX11_ABI=0'
    ],
    language='c++'  # 明确指定 C++ 语言
)

setup(
    name="grid_subsampling",
    version="0.1",
    ext_modules=[module]
)