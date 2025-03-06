#!/bin/bash
# Function to add paths to LD_LIBRARY_PATH without duplicates
add_to_path() {
    local new_paths="$1"
    local current_paths
    local path_to_add
    
    # Split current LD_LIBRARY_PATH into an array
    IFS=':' read -ra current_paths <<< "${LD_LIBRARY_PATH:-}"
    
    # Split new paths into an array
    IFS=':' read -ra paths_to_add <<< "$new_paths"
    
    # Process each new path
    for path_to_add in "${paths_to_add[@]}"; do
        # Skip empty paths
        [ -z "$path_to_add" ] && continue
        
        # Remove trailing slash if present
        path_to_add="${path_to_add%/}"
        
        # Check if path already exists
        local is_duplicate=false
        for existing_path in "${current_paths[@]}"; do
            existing_path="${existing_path%/}"
            if [ "$existing_path" = "$path_to_add" ]; then
                echo -e "\033[33m⚠\033[0m \033[1mWARN\033[0m: Path already exists in LD_LIBRARY_PATH: $path_to_add"
                is_duplicate=true
                break
            fi
        done
        
        # Add path if it's not a duplicate
        if [ "$is_duplicate" = false ]; then
            if [ -z "$LD_LIBRARY_PATH" ]; then
                LD_LIBRARY_PATH="$path_to_add"
            else
                LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$path_to_add"
            fi
            echo -e "\033[34m🛈\033[0m \033[1mINFO\033[0m: Added to LD_LIBRARY_PATH: $path_to_add"
        fi
    done
}

# Ensure we are inside a virtual environment
if [ -z "$VIRTUAL_ENV" ]; then
    echo -e "\033[31m✘\033[0m \033[1mERROR\033[0m: Not inside a virtual environment. Please activate one first."
    exit 1
fi

# Print the current virtual environment 
echo -e "\033[34m🛈\033[0m \033[1mINFO\033[0m: Current environment: $VIRTUAL_ENV"

# Add custom library paths
add_to_path "$VIRTUAL_ENV/lib"

# Try to locate TensorRT path
TENSORRT_LIBS_PATH=$(dirname $(python -c "import tensorrt_libs;print(tensorrt_libs.__file__)" 2>/dev/null) 2>/dev/null)
if [ -d "$TENSORRT_LIBS_PATH" ]; then
    echo -e "\033[34m🛈\033[0m \033[1mINFO\033[0m: Found TensorRT path: $TENSORRT_LIBS_PATH"
    add_to_path "$TENSORRT_LIBS_PATH"
else
    echo -e "\033[33m⚠\033[0m \033[1mWARN\033[0m: TensorRT path not found, not adding it to LD_LIBRARY_PATH"
fi

# Try to locate CUDA path
CUDA_PATH="/usr/local/cuda/lib64"
if [ -d "$CUDA_PATH" ]; then
    echo -e "\033[34m🛈\033[0m \033[1mINFO\033[0m: Found CUDA path: $CUDA_PATH"
    add_to_path "$CUDA_PATH"
else
    echo -e "\033[33m⚠\033[0m \033[1mWARN\033[0m: CUDA path $CUDA_PATH not found, not adding it to LD_LIBRARY_PATH"
fi

# Try to locate NVIDIA/cuda paths
NVIDIA_DIR=$(dirname $(dirname $(python -c "import nvidia.cudnn;print(nvidia.cudnn.__file__)" 2>/dev/null)) 2>/dev/null)
if [ -d "$NVIDIA_DIR" ]; then
    CUDNN_PATHS=$(echo ${NVIDIA_DIR}/*/lib/ | sed -r 's/\s+/:/g')
    echo -e "\033[34m🛈\033[0m \033[1mINFO\033[0m: Found CUDA paths: $CUDNN_PATHS"
    add_to_path "$CUDNN_PATHS"
else
    echo -e "\033[33m⚠\033[0m \033[1mWARN\033[0m: nvidia/cuda, cuDNN, cuFFT, etc. paths not found, not adding it to LD_LIBRARY_PATH"
fi

export LD_LIBRARY_PATH="$LD_LIBRARY_PATH"
echo -e "\033[34m🛈\033[0m \033[1mINFO\033[0m: New LD_LIBRARY_PATH: $LD_LIBRARY_PATH"
