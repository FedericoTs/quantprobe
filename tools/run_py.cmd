@echo off
REM Run any python module/args in the portable CUDA + MSVC build env.
REM Usage: run_py.cmd weights.evoq_run gate
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul
set "CUDA_HOME=C:\Users\Federico\Documents\evo-compress\tools\cuda_portable"
set "CUDA_PATH=%CUDA_HOME%"
set "PATH=%CUDA_HOME%\bin;%PATH%"
set "TORCH_CUDA_ARCH_LIST=6.1"
cd /d C:\Users\Federico\Documents\evo-compress
.venv\Scripts\python.exe -m %*
