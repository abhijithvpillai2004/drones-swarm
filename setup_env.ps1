# One-time setup on a new Windows machine (run from the project folder):
#   powershell -ExecutionPolicy Bypass -File setup_env.ps1
#
# Uses conda if it is installed (prebuilt pybullet, fastest). Otherwise creates
# a local venv and installs with pip, which compiles pybullet from source and
# needs "Visual Studio Build Tools" with the C++ workload (10-20 minutes).
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

# gym-pybullet-drones is cloned into external/ (pendrive copies may already have it)
if (-not (Test-Path "external/gym-pybullet-drones")) {
    New-Item -ItemType Directory -Force external | Out-Null
    git clone https://github.com/utiasDSL/gym-pybullet-drones.git external/gym-pybullet-drones
}

if (Get-Command conda -ErrorAction SilentlyContinue) {
    Write-Host "[setup] conda found: creating env 'downwash_swarm' from environment.yml"
    conda env create -f environment.yml
    conda run -n downwash_swarm pip install -e external/gym-pybullet-drones --no-deps
    Write-Host "[setup] done. Next:  conda activate downwash_swarm ; python -m pytest -q"
} else {
    Write-Host "[setup] conda not found: creating .\venv with pip"
    python -m venv venv
    .\venv\Scripts\python -m pip install --upgrade pip
    .\venv\Scripts\python -m pip install -r requirements.txt
    .\venv\Scripts\python -m pip install -e external/gym-pybullet-drones --no-deps
    .\venv\Scripts\python -m pip install gymnasium transforms3d pillow
    Write-Host "[setup] done. Next:  .\venv\Scripts\Activate.ps1 ; python -m pytest -q"
}
