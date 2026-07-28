@echo off
REM Portable, no-admin CUDA build env (MSVC host + extracted CUDA 12.4) -> run trellis_run.
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul
set "CUDA_HOME=<repo>\tools\cuda_portable"
set "CUDA_PATH=%CUDA_HOME%"
set "PATH=%CUDA_HOME%\bin;%PATH%"
set "TORCH_CUDA_ARCH_LIST=6.1"
cd /d <repo>
.venv\Scripts\python.exe -u -m weights.trellis_run %*
