
function Check-CUDA {
    $ver = "12.9"
    
    $cudaPath = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v$ver"
    if (-not (Test-Path $cudaPath)) {
        Write-Host "ERROR: CUDA v$ver not found at $cudaPath!" -ForegroundColor Red
        Write-Host "Please install the CUDA Toolkit before running this build." -ForegroundColor Yellow
        exit 1
    } else {
        Write-Host "Success: CUDA v$ver detected." -ForegroundColor Green
    }
}

function Get-OpenVINO {
    param ([string]$destination)

    $url = "https://storage.openvinotoolkit.org/repositories/openvino/packages/2025.4.2/windows/openvino_toolkit_windows_2025.4.2.20430.85e49f27be1_x86_64.zip"
    
    $zipPath = Join-Path $destination "OpenVINO.zip"
    $extractPath = Join-Path $destination "OpenVINO"

    if (-not (Test-Path $extractPath)) {
        Write-Host "Downloading OpenVINO Toolkit for x64..." -ForegroundColor Magenta
        New-Item -ItemType Directory -Path $destination -Force | Out-Null
        Invoke-WebRequest -Uri $url -OutFile $zipPath
        Write-Host "Unzipping OpenVINO (This may take a while)..."
        Expand-Archive -Path $zipPath -DestinationPath $extractPath -Force
        Remove-Item $zipPath
    } else {
        Write-Host "OpenVINO for x64 already exists." -ForegroundColor DarkGray
    }
    
    # Return the path to the actual extracted toolkit folder inside
    $toolkitFolder = Get-ChildItem -Path $extractPath -Directory | Select-Object -First 1
    return $toolkitFolder.FullName
}

function Get-CuDNN {
    param ([string]$destination)
    
    # Versions for cuDNN (Specific to 9.19.1 for CUDA 12)
    $cudnnVer = "9.19.1.2"
    $cudaTarget = "cuda12"

    $url = "https://developer.download.nvidia.com/compute/cudnn/redist/cudnn/windows-x86_64/cudnn-windows-x86_64-${cudnnVer}_${cudaTarget}-archive.zip"
    
    $zipPath = Join-Path $destination "cuDNN.zip"
    $extractPath = Join-Path $destination "cuDNN"

    if (-not (Test-Path $extractPath)) {
        Write-Host "Downloading cuDNN v$cudnnVer for x64..." -ForegroundColor Magenta
        New-Item -ItemType Directory -Path $destination -Force | Out-Null
        Invoke-WebRequest -Uri $url -OutFile $zipPath
        Write-Host "Unzipping cuDNN..."
        Expand-Archive -Path $zipPath -DestinationPath $extractPath -Force
        Remove-Item $zipPath
    } else {
        Write-Host "cuDNN v$cudnnVer for x64 already exists." -ForegroundColor DarkGray
    }

    # Return the path to the actual extracted cudnn folder inside
    $cudnnFolder = Get-ChildItem -Path $extractPath -Directory | Select-Object -First 1
    return $cudnnFolder.FullName
}

# === Build Configurations ===

function Configure {
    param (
        [string]$opencvSource = "opencv",         
        [string]$opencvContribSource = "opencv_contrib", 
        [string]$cmakeGenerator = "Visual Studio 17 2022",
        [string]$arch,
        [string]$buildDir,
        [bool]$enableCuda = $false
    )

    $opencvAbsPath = (Resolve-Path "$PSScriptRoot\$opencvSource").Path.Replace('\', '/')
    $contribAbsPath = (Resolve-Path "$PSScriptRoot\$opencvContribSource\modules").Path.Replace('\', '/')
    
    $depDir = Join-Path $PSScriptRoot "deps_$arch"

    $fullBuildDir = Join-Path $PSScriptRoot $buildDir
    if (!(Test-Path $fullBuildDir)) {
        New-Item -ItemType Directory -Path $fullBuildDir | Out-Null
    }

    Push-Location $fullBuildDir

    if ($arch -eq "x64") {
        $dispatchFlags = "AVX,AVX2,AVX512_SKX"
    } else {
        $dispatchFlags = "AVX"
    }

    # Base CMake Arguments
    $cmakeArgs = @(
        "-G", "$cmakeGenerator",
        "-A", $arch,
        "-D", "CMAKE_CONFIGURATION_TYPES=Release",
        "-D", "BUILD_SHARED_LIBS=ON",
        "-D", "BUILD_opencv_world=ON",
        "-D", "BUILD_TESTS=OFF",
        "-D", "BUILD_PERF_TESTS=OFF",
        "-D", "CMAKE_INSTALL_PREFIX=$fullBuildDir/install",
        "-D", "BUILD_opencv_python2=OFF",
        "-D", "BUILD_opencv_python3=OFF",
        "-D", "BUILD_opencv_python_bindings_generator=OFF",
        "-D", "BUILD_opencv_python_tests=OFF",
        "-D", "BUILD_JAVA=OFF",
        "-D", "BUILD_opencv_java_bindings_generator=OFF",
        "-D", "PYTHON_EXECUTABLE=NONEXISTENT",
        "-D", "PYTHON3_EXECUTABLE=NONEXISTENT",
        "-D", "CPU_BASELINE=SSE3",
        "-D", "CPU_DISPATCH=$dispatchFlags"
    )

    if ($arch -eq "x64") {
        # Get OpenVINO for x64
        $openVinoPath = Get-OpenVINO -destination $depDir
        $openVinoPathCmake = $openVinoPath.Replace('\', '/')
        
        Write-Host "Configuring OpenVINO for x64..." -ForegroundColor Cyan
        $cmakeArgs += @(
            "-D", "WITH_OPENVINO=ON",
            "-D", "OpenVINO_DIR=$openVinoPathCmake/runtime/cmake"
        )

        if ($enableCuda) {
            # Get cuDNN locally instead of using global program files
            $cudnnPath = Get-CuDNN -destination $depDir
            $cudnnPathCmake = $cudnnPath.Replace('\', '/')

            Write-Host "Configuring with CUDA & cuDNN support for x64..." -ForegroundColor Cyan
            $cmakeArgs += @(
                "-D", "OPENCV_EXTRA_MODULES_PATH=$contribAbsPath",
                "-D", "WITH_CUDA=ON",
                "-D", "WITH_CUDNN=ON",
                "-D", "OPENCV_DNN_CUDA=ON",
                "-D", "CUDNN_INCLUDE_DIR=$cudnnPathCmake/include",
                "-D", "CUDNN_LIBRARY=$cudnnPathCmake/lib/x64/cudnn.lib",
                "-D", "CUDA_ARCH_BIN=6.1;7.5;8.6;8.9;12.0",
                "-D", "CUDA_ARCH_PTX=8.9" 
            )
        }
    } else {
        Write-Host "Configuring CPU-Only build for $arch..." -ForegroundColor Yellow
        $cmakeArgs += @("-D", "WITH_CUDA=OFF", "-D", "WITH_OPENVINO=OFF")
    }

    $cmakeArgs += $opencvAbsPath

    & cmake $cmakeArgs

    if ($LASTEXITCODE -eq 0) {
        Write-Host "CMake Configuration Succeeded." -ForegroundColor Green
    } else {
        Write-Host "CMake Configuration Failed! Skipping Build." -ForegroundColor Red
    }

    Pop-Location
}

function Build {
    param (
        [string]$buildDir
    )

    $fullBuildDir = Join-Path $PSScriptRoot $buildDir
    Push-Location $fullBuildDir
    
    cmake --build . --config Release --target INSTALL
    
    Pop-Location
}

# === Execution ===

Check-CUDA

# === x64 (WITH CUDA & OPENVINO) ===
Configure -arch "x64" -buildDir "Build_OpenCvWorld_x64" -enableCuda $true
Build -buildDir "Build_OpenCvWorld_x64"

# === Win32 (CPU ONLY) ===
Configure -arch "Win32" -buildDir "Build_OpenCvWorld_x86" -enableCuda $false
Build -buildDir "Build_OpenCvWorld_x86"