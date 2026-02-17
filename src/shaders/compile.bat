:: ============================================================
:: File: shaders/compile.bat
:: Role: GLSL compute shader → SPIR-V 컴파일
:: Usage: 더블클릭 또는 cmd에서 실행
:: ============================================================
@echo off

:: Vulkan SDK의 glslc 컴파일러 사용
:: glslc: Google 제공, glslangValidator보다 에러 메시지 친절
::
:: 옵션:
::   -fshader-stage=comp  : compute shader 명시
::   -o output.spv        : 출력 파일명

echo Compiling shaders...

glslc simple.comp -o simple.spv

if %errorlevel% neq 0 (
    echo [ERROR] Shader compilation failed!
    pause
    exit /b 1
)

echo [OK] simple.comp → simple.spv
pause
