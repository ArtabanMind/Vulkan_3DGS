cmake_minimum_required(VERSION 3.20)
project(gaussian_splat LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)

# Vulkan
find_package(Vulkan REQUIRED)

# GLFW (수동)
set(GLFW_DIR ${CMAKE_SOURCE_DIR}/third_party/glfw)
add_library(glfw STATIC IMPORTED)
set_target_properties(glfw PROPERTIES
    IMPORTED_LOCATION ${GLFW_DIR}/lib-vc2022/glfw3.lib
    INTERFACE_INCLUDE_DIRECTORIES ${GLFW_DIR}/include
)

# GLM (헤더 온리)
set(GLM_DIR ${CMAKE_SOURCE_DIR}/third_party/glm)

add_executable(${PROJECT_NAME}
    src/main.cpp
)

target_include_directories(${PROJECT_NAME} PRIVATE
    src/
    ${GLM_DIR}
)

target_link_libraries(${PROJECT_NAME}
    Vulkan::Vulkan
    glfw
)