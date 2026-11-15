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

add_executable(i18n_font_test i18n_font_test.cpp ../theme.cpp
    "${IMGUI_SOURCE_DIR}/imgui.cpp" "${IMGUI_SOURCE_DIR}/imgui_draw.cpp"
    "${IMGUI_SOURCE_DIR}/imgui_tables.cpp" "${IMGUI_SOURCE_DIR}/imgui_widgets.cpp")
target_include_directories(i18n_font_test PRIVATE .. "${IMGUI_SOURCE_DIR}")
target_compile_features(i18n_font_test PRIVATE cxx_std_20)
add_test(NAME i18n_fonts COMMAND i18n_font_test)

find_package(Python3 COMPONENTS Interpreter QUIET)
if(Python3_Interpreter_FOUND)
    add_test(NAME i18n_catalog COMMAND "${Python3_EXECUTABLE}"
        "${CMAKE_CURRENT_SOURCE_DIR}/i18n_catalog_test.py")
endif()
if(MSVC)
    target_compile_options(i18n_font_test PRIVATE /utf-8)
endif()
