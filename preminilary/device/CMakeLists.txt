cmake_minimum_required(VERSION 3.0.0)

project(ssne_ai_demo)

include(cmake_config/Paths.cmake)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O2")
add_compile_options(-Wno-psabi)

add_executable(ssne_ai_demo ${PROJECT_SOURCE_DIR}/main.cpp)

include_directories(${M1_SDK_INC_DIR})
include_directories(${M1_SDK_INC_DIR}/smartsoc)
include_directories(${CMAKE_SOURCE_DIR}/include)

file(GLOB SRC_FILES "${CMAKE_SOURCE_DIR}/src/*.cpp")
target_sources(ssne_ai_demo PRIVATE ${SRC_FILES})

target_link_libraries(${PROJECT_NAME}
    ${M1_SSNE_LIB}
    ${M1_CMABUFFER_LIB}
    ${M1_OSD_LIB}
    ${M1_SSZLOG_LIB}
    ${M1_ZLOG_LIB}
    ${M1_EMB_LIB}
)

install(TARGETS ssne_ai_demo DESTINATION bin)
