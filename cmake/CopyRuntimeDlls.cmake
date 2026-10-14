# TARGET_RUNTIME_DLLS can be empty for a statically linked executable.
string(REPLACE "|" ";" _photara_runtime_files "${PHOTARA_RUNTIME_DLLS}")
foreach(_photara_runtime_file IN LISTS _photara_runtime_files)
    get_filename_component(_photara_runtime_name "${_photara_runtime_file}" NAME)
    file(COPY_FILE "${_photara_runtime_file}"
        "${PHOTARA_RUNTIME_DESTINATION}/${_photara_runtime_name}" ONLY_IF_DIFFERENT)
endforeach()
