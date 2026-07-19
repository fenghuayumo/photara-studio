# Intrinsic delighter model directory (ONNX Runtime, C++ only).
# Place stage_0.onnx .. stage_3.onnx under this path.
# License: academic / non-commercial (compphoto/Intrinsic). See docs/LICENSE-Intrinsic.md.

function(aetherscan_setup_intrinsic_models)
    set(AETHERSCAN_INTRINSIC_MODELS_DIR
        "${CMAKE_BINARY_DIR}/Models/Intrinsic"
        CACHE PATH "Directory containing Intrinsic stage_*.onnx models")
endfunction()
