# PowerShell Script to build OpenCV modules and opencv_world in two steps
function Get-OpenBLAS {
    param (
        [string]$arch,
        [string]$destination
    )

    # Version OpenBLAS
    $ver = "0.3.30"
    
 
    if ($arch -eq "x64") {
        $url = "https://github.com/OpenMathLib/OpenBLAS/releases/download/v$ver/OpenBLAS-$ver-x64.zip"
    } else {
        $url = "https://github.com/OpenMathLib/OpenBLAS/releases/download/v$ver/OpenBLAS-$ver-x86.zip"
    }

    $zipPath = Join-Path $destination "OpenBLAS.zip"
    $extractPath = Join-Path $destination "OpenBLAS"

    if (-not (Test-Path $extractPath)) {
        Write-Host "Download OpenBLAS for $arch " -ForegroundColor Magenta
        New-Item -ItemType Directory -Path $destination -Force | Out-Null
        
        Invoke-WebRequest -Uri $url -OutFile $zipPath
 
        Write-Host "Unzip OpenBLAS..."
        Expand-Archive -Path $zipPath -DestinationPath $extractPath -Force
        
        Remove-Item $zipPath
    } else {
        Write-Host "OpenBLAS for $arch allready exists." -ForegroundColor DarkGray
    }

    return $extractPath
}

function Configure {
    param (
        [string]$opencvSource = "opencv",
        [string]$cmakeGenerator = "Visual Studio 17 2022",
        [string]$arch,
        [string]$buildDir
    )

    $depDir = Join-Path $PWD "deps_$arch"
    $openBlasPath = Get-OpenBLAS -arch $arch -destination $depDir   
    $openBlasPathCmake = $openBlasPath.Replace('\', '/')

    if (!(Test-Path $buildDir)) {
        New-Item -ItemType Directory -Path $buildDir | Out-Null
    }

    Push-Location $buildDir

    if ($arch -eq "x64") {
        $dispatchFlags = "AVX,AVX2,AVX512_SKX"
    }
    else {
        $dispatchFlags = "AVX"
    }

    cmake `
        -G "$cmakeGenerator" `
        -A $arch `
        -D CMAKE_CONFIGURATION_TYPES=Release `
        -D BUILD_SHARED_LIBS=ON `
        -D BUILD_opencv_world=ON `
        -D CMAKE_INSTALL_PREFIX="$PWD/install" `
        -D BUILD_opencv_python2=OFF `
        -D BUILD_opencv_python3=OFF `
        -D BUILD_opencv_python_bindings_generator=OFF `
        -D BUILD_opencv_python_tests=OFF `
        -D BUILD_JAVA=OFF `
        -D BUILD_opencv_java_bindings_generator=OFF `
        -D PYTHON_EXECUTABLE="NONEXISTENT" `
        -D PYTHON3_EXECUTABLE="NONEXISTENT" `
        -D CPU_BASELINE=SSE3 `
        -D CPU_DISPATCH=$dispatchFlags `
        -D WITH_OPENBLAS=ON `
        -D WITH_LAPACK=ON `
        -D OpenBLAS_INCLUDE_DIR=$openBlasPathCmake/include `
        -D OpenBLAS_LIB=$openBlasPathCmake/lib/libopenblas.lib `
        "..\$opencvSource"

    cmake --build . --config Release --target INSTALL

    if ($LASTEXITCODE -eq 0) {
        cmake --build . --config Release --target INSTALL
    } else {
        Write-Host "CMake Configuration Failed! Skipping Build." -ForegroundColor Red
    }

    Pop-Location
}

function Build {
    param (
        [string]$buildDir
    )

    Push-Location $buildDir
    cmake --build . --config Release
    Pop-Location
}

# === x64 ===
Configure -arch "x64" -buildDir "Build_OpenCvWorld_x64"
Build -buildDir "Build_OpenCvWorld_x64"

# === Win32 ===
Configure -arch "Win32" -buildDir "Build_OpenCvWorld_x86"
Build -buildDir "Build_OpenCvWorld_x86"