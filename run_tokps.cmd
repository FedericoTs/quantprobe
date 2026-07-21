@echo off
REM ============================================================
REM  tok/s benchmark: 16B MoE (DeepSeek-Coder-V2-Lite) at 2-bit
REM  on the GTX 1060 6GB.
REM
REM  RUN THIS WITH CLAUDE, BROWSERS, AND THE NVIDIA OVERLAY CLOSED
REM  so the GPU has maximum free VRAM for the model.
REM
REM  How:  close those apps, then double-click this file
REM        (or run it from a cmd window). It takes ~1-2 min.
REM  After it finishes, reopen Claude and say:
REM        "read tokps_result.log"
REM ============================================================
setlocal
set LLAMA=D:\evo-compress-data\llamacpp
set GGUF=D:\evo-compress-data\gguf\DeepSeek-Coder-V2-Lite-Base-IQ2_XS.gguf
set OUT=C:\Users\Federico\Documents\evo-compress\weights\data\tokps_result.log

echo ===== tok/s benchmark (run with Claude/apps CLOSED) =====> "%OUT%"
echo.>> "%OUT%"
echo --- free VRAM before load (more free = more of the model on GPU) --->> "%OUT%"
nvidia-smi --query-gpu=memory.used,memory.free,memory.total --format=csv,noheader>> "%OUT%" 2>&1
echo.>> "%OUT%"
echo --- llama-bench: IQ2_XS, full GPU offload (-ngl 99), 128-token generation --->> "%OUT%"
"%LLAMA%\llama-bench.exe" -m "%GGUF%" -ngl 99 -p 0 -n 128>> "%OUT%" 2>&1
echo.>> "%OUT%"
echo ===== DONE -- reopen Claude and say: read tokps_result.log =====>> "%OUT%"

echo.
echo ============================================================
echo  DONE. Results written to:
echo    %OUT%
echo  Now reopen Claude and say:  read tokps_result.log
echo ============================================================
echo.
pause
