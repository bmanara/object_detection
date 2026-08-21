cmake_minimum_required(VERSION 3.20)
project(object_detection)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(image_transport REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS dnn)

list(APPEND CMAKE_PREFIX_PATH "/home/bmacraze/libtorch")
find_package(Torch REQUIRED)

set(CMAKE_BUILD_RPATH "/home/bmacraze/libtorch/lib")
set(CMAKE_INSTALL_RPATH "/home/bmacraze/libtorch/lib")
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

message(STATUS "Torch library: ${TORCH_LIBRARIES}")

# Bring in the local tokenizer-cpp library from the home directory.
add_subdirectory($ENV{HOME}/tokenizers-cpp tokenizers-cpp EXCLUDE_FROM_ALL)
# uncomment the following section in order to fill in
# further dependencies manually.
# find_package(<dependency> REQUIRED)

# ONNX Runtime Path
if (NOT DEFINED ONNXRUNTIME_ROOTDIR)
  set(ONNXRUNTIME_ROOTDIR "$ENV{HOME}/onnxruntime")
endif()

message(STATUS "ONNXRUNTIME_ROOTDIR: ${ONNXRUNTIME_ROOTDIR}")

include_directories(${ONNXRUNTIME_ROOTDIR}/include)
find_library(ONNXRUNTIME_LIB onnxruntime HINTS ${ONNXRUNTIME_ROOTDIR}/lib)
if(NOT ONNXRUNTIME_LIB)
  message(FATAL_ERROR "onnxruntime library not found under ${ONNXRUNTIME_ROOTDIR}/lib")
endif()

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

add_executable(image_object_detector src/image_object_detector.cpp)
target_link_libraries(image_object_detector PUBLIC
  rclcpp::rclcpp
  sensor_msgs::sensor_msgs_library
  cv_bridge::cv_bridge
  image_transport::image_transport
  opencv_dnn
  ${ONNXRUNTIME_LIB}
  ${TORCH_LIBRARIES}
)

target_link_libraries(image_object_detector PRIVATE tokenizers_cpp)

add_executable(video_publisher_node src/video_publisher_node.cpp)
target_link_libraries(video_publisher_node PUBLIC
  rclcpp::rclcpp
  sensor_msgs::sensor_msgs_library
  cv_bridge::cv_bridge
  image_transport::image_transport
  opencv_dnn
  ${OpenCV_LIBS}
)

install(TARGETS image_object_detector video_publisher_node
  RUNTIME DESTINATION lib/${PROJECT_NAME}
)

ament_package()
