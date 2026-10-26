cmake_minimum_required(VERSION 3.28)
# Fetch only build sources, excluding test fixtures containing Windows symlinks.
set(_archive "${SOURCE_DIR}/lightusd.tar.gz")
file(MAKE_DIRECTORY "${SOURCE_DIR}")
file(DOWNLOAD
    "https://codeload.github.com/lighttransport/tinyusdz/tar.gz/d1ede90f919f6fe7c0014487be5441afb919cc79"
    "${_archive}" STATUS _status TLS_VERIFY ON
    EXPECTED_HASH SHA256=e7a64a7b61e3f64506436c2328fa2f59c85f983012f82ec07fcf880432459f58)
list(GET _status 0 _code)
if(NOT _code EQUAL 0)
    message(FATAL_ERROR "LightUSD download failed: ${_status}")
endif()
set(_prefix "LightUSD-d1ede90f919f6fe7c0014487be5441afb919cc79")
# libarchive attempts locale conversion even for excluded fixture names. On
# Chinese Windows (CP936) that can abort before the source-only filter runs.
# Read the archive's UTF-8 names with Python and write only regular build files.
find_program(_python NAMES python3 python REQUIRED)
execute_process(COMMAND "${_python}" "${CMAKE_CURRENT_LIST_DIR}/ExtractLightUsd.py"
    "${_archive}" "${SOURCE_DIR}" "${_prefix}"
    COMMAND_ERROR_IS_FATAL ANY)
