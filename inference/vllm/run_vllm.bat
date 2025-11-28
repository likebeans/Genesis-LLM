@echo off
REM vLLM Docker 启动脚本 (Windows)
REM 用法: run_vllm.bat [模型路径或HF模型名]

setlocal enabledelayedexpansion

REM ========== 配置 ==========
set MODEL=%~1
if "%MODEL%"=="" set MODEL=Qwen/Qwen2.5-0.5B-Instruct

set PORT=8000
set GPU_MEMORY_UTILIZATION=0.9
set MAX_MODEL_LEN=4096
set TENSOR_PARALLEL_SIZE=1

REM ========== HuggingFace 缓存 ==========
if "%HF_HOME%"=="" (
    set HF_CACHE=%USERPROFILE%\.cache\huggingface
) else (
    set HF_CACHE=%HF_HOME%
)
if not exist "%HF_CACHE%" mkdir "%HF_CACHE%"

REM ========== 判断模型来源 ==========
if exist "%MODEL%\" (
    REM 本地模型目录
    set MODEL_PATH=%MODEL%
    set MOUNT_ARGS=-v "%MODEL_PATH%:/model"
    set MODEL_ARG=/model
    echo [Local] Using local model: %MODEL_PATH%
) else (
    REM HuggingFace 模型
    set MOUNT_ARGS=
    set MODEL_ARG=%MODEL%
    echo [HF] Using HuggingFace model: %MODEL%
)

REM ========== 启动容器 ==========
echo.
echo Starting vLLM service...
echo   Port: %PORT%
echo   GPU Memory: %GPU_MEMORY_UTILIZATION%
echo   Max Context: %MAX_MODEL_LEN%
echo.

docker run --rm -it ^
    --gpus all ^
    -p %PORT%:8000 ^
    -v "%HF_CACHE%:/root/.cache/huggingface" ^
    %MOUNT_ARGS% ^
    --ipc=host ^
    vllm/vllm-openai:latest ^
    --model "%MODEL_ARG%" ^
    --served-model-name "%MODEL%" ^
    --tensor-parallel-size %TENSOR_PARALLEL_SIZE% ^
    --gpu-memory-utilization %GPU_MEMORY_UTILIZATION% ^
    --max-model-len %MAX_MODEL_LEN% ^
    --trust-remote-code

echo.
echo vLLM service started
echo   API: http://localhost:%PORT%/v1
echo   Docs: http://localhost:%PORT%/docs

endlocal
