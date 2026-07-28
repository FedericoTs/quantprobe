@echo off
REM ============================================================================
REM  Pre-registration #61: cold-boot A/B - run this FIRST after a fresh reboot,
REM  before opening anything else. Takes ~25 min. Results append to:
REM    weights\data\prereg61_coldboot.log
REM  Arms (most important first, in case of interruption):
REM    1. instrumented binary, tool -ot split  (direct #60 comparison)
REM    2. PRISTINE binary (zero patches),  same arm  (fair-binary check)
REM    3. instrumented, plain -ngl 20             4. pristine, plain -ngl 20
REM    5. pristine, -ot split pp2048              (the original 386 pp claim)
REM  GPU clocks sampled before/after every arm - the #25 discipline.
REM ============================================================================
setlocal
set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9"
set "PATH=%CUDA_PATH%\bin;C:\Windows\System32;%PATH%"
set "LOG=%~dp0data\prereg61_coldboot.log"
set "INSTR=C:\Users\Federico\Documents\evo-compress\tools\llama.cpp-src\build-cuda\bin\llama-bench.exe"
set "PRIST=C:\Users\Federico\Documents\evo-compress\tools\llama.cpp-pristine\build\bin\llama-bench.exe"
set "MODEL=D:/evo-compress-data/gguf/_k_sweep_scratch.gguf"
set "OT=blk\.(1[1-9]|[2-3][0-9]|4[0-7])\.ffn_.*_exps\.=CPU"

echo ================================================================ >> "%LOG%"
echo === COLD BOOT RUN %DATE% %TIME% === >> "%LOG%"
nvidia-smi --query-gpu=clocks.sm,clocks.mem,temperature.gpu,power.draw --format=csv,noheader >> "%LOG%"

echo --- arm 1: INSTRUMENTED, tool -ot split, tg128 --- >> "%LOG%"
"%INSTR%" -m "%MODEL%" -ngl 99 -ot "%OT%" -mmp 0 -b 1024 -ub 1024 -n 128 -p 0 -r 2 2>nul | findstr tg128 >> "%LOG%"
nvidia-smi --query-gpu=clocks.sm,clocks.mem,temperature.gpu --format=csv,noheader >> "%LOG%"

echo --- arm 2: PRISTINE, tool -ot split, tg128 --- >> "%LOG%"
"%PRIST%" -m "%MODEL%" -ngl 99 -ot "%OT%" -mmp 0 -b 1024 -ub 1024 -n 128 -p 0 -r 2 2>nul | findstr tg128 >> "%LOG%"
nvidia-smi --query-gpu=clocks.sm,clocks.mem,temperature.gpu --format=csv,noheader >> "%LOG%"

echo --- arm 3: INSTRUMENTED, plain -ngl 20, tg128 --- >> "%LOG%"
"%INSTR%" -m "%MODEL%" -ngl 20 -mmp 0 -n 128 -p 0 -r 2 2>nul | findstr tg128 >> "%LOG%"
nvidia-smi --query-gpu=clocks.sm,temperature.gpu --format=csv,noheader >> "%LOG%"

echo --- arm 4: PRISTINE, plain -ngl 20, tg128 --- >> "%LOG%"
"%PRIST%" -m "%MODEL%" -ngl 20 -mmp 0 -n 128 -p 0 -r 2 2>nul | findstr tg128 >> "%LOG%"
nvidia-smi --query-gpu=clocks.sm,temperature.gpu --format=csv,noheader >> "%LOG%"

echo --- arm 5: PRISTINE, -ot split, pp2048 (the original 386 claim) --- >> "%LOG%"
"%PRIST%" -m "%MODEL%" -ngl 99 -ot "%OT%" -mmp 0 -b 1024 -ub 1024 -n 0 -p 2048 -r 2 2>nul | findstr pp2048 >> "%LOG%"
nvidia-smi --query-gpu=clocks.sm,clocks.mem,temperature.gpu --format=csv,noheader >> "%LOG%"

echo --- position control: repeat arm 1 LAST --- >> "%LOG%"
"%INSTR%" -m "%MODEL%" -ngl 99 -ot "%OT%" -mmp 0 -b 1024 -ub 1024 -n 128 -p 0 -r 2 2>nul | findstr tg128 >> "%LOG%"
nvidia-smi --query-gpu=clocks.sm,clocks.mem,temperature.gpu --format=csv,noheader >> "%LOG%"

echo === DONE %TIME% === >> "%LOG%"
echo.
echo Cold-boot A/B complete. Results in weights\data\prereg61_coldboot.log
pause
