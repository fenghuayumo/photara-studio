cmake_minimum_required(VERSION 3.24)
project(photara_shortcuts_test LANGUAGES CXX)
# Standalone: configure with -DIMGUI_SOURCE_DIR=<matching Dear ImGui sources>.
if(NOT EXISTS "${IMGUI_SOURCE_DIR}/imgui.cpp")
    message(FATAL_ERROR "Set IMGUI_SOURCE_DIR to the Studio Dear ImGui source directory")
endif()
add_executable(shortcuts_test shortcuts_test.cpp
    "${IMGUI_SOURCE_DIR}/imgui.cpp" "${IMGUI_SOURCE_DIR}/imgui_draw.cpp"
    "${IMGUI_SOURCE_DIR}/imgui_tables.cpp" "${IMGUI_SOURCE_DIR}/imgui_widgets.cpp")
target_include_directories(shortcuts_test PRIVATE .. "${IMGUI_SOURCE_DIR}")
target_compile_features(shortcuts_test PRIVATE cxx_std_20)
enable_testing()
add_test(NAME shortcuts COMMAND shortcuts_test)
