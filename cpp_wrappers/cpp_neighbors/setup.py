from setuptools import setup, Extension
import numpy as np  # 更简洁的 numpy 头文件引入方式

# 源文件路径保持不变
SOURCES = [
    "../cpp_utils/cloud/cloud.cpp",
    "neighbors/neighbors.cpp",
    "wrapper.cpp"
]

# 重构 Extension 配置
module = Extension(
    name="radius_neighbors",
    sources=SOURCES,
    include_dirs=[np.get_include()],  # 将 numpy 头文件路径移入 Extension
    extra_compile_args=[
        '-std=c++11',                # 保持编译参数不变
        '-D_GLIBCXX_USE_CXX11_ABI=0'
    ],
    language='c++'                   # 显式指定 C++ 语言
)

# 优化 setup 配置
setup(
    name="radius_neighbors",         # 添加包名称
    version="0.1.0",                 # 添加版本号
    ext_modules=[module],            # 保持扩展模块配置
    install_requires=["numpy"],      # 显式声明依赖
    zip_safe=False                   # 禁用 zip 安装确保 C++ 扩展正常工作
)






