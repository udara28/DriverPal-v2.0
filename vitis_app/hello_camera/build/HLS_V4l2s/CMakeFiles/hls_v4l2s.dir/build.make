# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.10

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /workspace

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /workspace/build

# Include any dependencies generated for this target.
include HLS_V4l2s/CMakeFiles/hls_v4l2s.dir/depend.make

# Include the progress variables for this target.
include HLS_V4l2s/CMakeFiles/hls_v4l2s.dir/progress.make

# Include the compile flags for this target's objects.
include HLS_V4l2s/CMakeFiles/hls_v4l2s.dir/flags.make

HLS_V4l2s/CMakeFiles/hls_v4l2s.dir/src/hlsV4l2Access.cpp.o: HLS_V4l2s/CMakeFiles/hls_v4l2s.dir/flags.make
HLS_V4l2s/CMakeFiles/hls_v4l2s.dir/src/hlsV4l2Access.cpp.o: ../HLS_V4l2s/src/hlsV4l2Access.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/workspace/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object HLS_V4l2s/CMakeFiles/hls_v4l2s.dir/src/hlsV4l2Access.cpp.o"
	cd /workspace/build/HLS_V4l2s && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/hls_v4l2s.dir/src/hlsV4l2Access.cpp.o -c /workspace/HLS_V4l2s/src/hlsV4l2Access.cpp

HLS_V4l2s/CMakeFiles/hls_v4l2s.dir/src/hlsV4l2Access.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/hls_v4l2s.dir/src/hlsV4l2Access.cpp.i"
	cd /workspace/build/HLS_V4l2s && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /workspace/HLS_V4l2s/src/hlsV4l2Access.cpp > CMakeFiles/hls_v4l2s.dir/src/hlsV4l2Access.cpp.i

HLS_V4l2s/CMakeFiles/hls_v4l2s.dir/src/hlsV4l2Access.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/hls_v4l2s.dir/src/hlsV4l2Access.cpp.s"
	cd /workspace/build/HLS_V4l2s && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /workspace/HLS_V4l2s/src/hlsV4l2Access.cpp -o CMakeFiles/hls_v4l2s.dir/src/hlsV4l2Access.cpp.s

HLS_V4l2s/CMakeFiles/hls_v4l2s.dir/src/hlsV4l2Access.cpp.o.requires:

.PHONY : HLS_V4l2s/CMakeFiles/hls_v4l2s.dir/src/hlsV4l2Access.cpp.o.requires

HLS_V4l2s/CMakeFiles/hls_v4l2s.dir/src/hlsV4l2Access.cpp.o.provides: HLS_V4l2s/CMakeFiles/hls_v4l2s.dir/src/hlsV4l2Access.cpp.o.requires
	$(MAKE) -f HLS_V4l2s/CMakeFiles/hls_v4l2s.dir/build.make HLS_V4l2s/CMakeFiles/hls_v4l2s.dir/src/hlsV4l2Access.cpp.o.provides.build
.PHONY : HLS_V4l2s/CMakeFiles/hls_v4l2s.dir/src/hlsV4l2Access.cpp.o.provides

HLS_V4l2s/CMakeFiles/hls_v4l2s.dir/src/hlsV4l2Access.cpp.o.provides.build: HLS_V4l2s/CMakeFiles/hls_v4l2s.dir/src/hlsV4l2Access.cpp.o


HLS_V4l2s/CMakeFiles/hls_v4l2s.dir/src/hlsV4l2Capture.cpp.o: HLS_V4l2s/CMakeFiles/hls_v4l2s.dir/flags.make
HLS_V4l2s/CMakeFiles/hls_v4l2s.dir/src/hlsV4l2Capture.cpp.o: ../HLS_V4l2s/src/hlsV4l2Capture.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/workspace/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object HLS_V4l2s/CMakeFiles/hls_v4l2s.dir/src/hlsV4l2Capture.cpp.o"
	cd /workspace/build/HLS_V4l2s && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/hls_v4l2s.dir/src/hlsV4l2Capture.cpp.o -c /workspace/HLS_V4l2s/src/hlsV4l2Capture.cpp

HLS_V4l2s/CMakeFiles/hls_v4l2s.dir/src/hlsV4l2Capture.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/hls_v4l2s.dir/src/hlsV4l2Capture.cpp.i"
	cd /workspace/build/HLS_V4l2s && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /workspace/HLS_V4l2s/src/hlsV4l2Capture.cpp > CMakeFiles/hls_v4l2s.dir/src/hlsV4l2Capture.cpp.i

HLS_V4l2s/CMakeFiles/hls_v4l2s.dir/src/hlsV4l2Capture.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/hls_v4l2s.dir/src/hlsV4l2Capture.cpp.s"
	cd /workspace/build/HLS_V4l2s && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /workspace/HLS_V4l2s/src/hlsV4l2Capture.cpp -o CMakeFiles/hls_v4l2s.dir/src/hlsV4l2Capture.cpp.s

HLS_V4l2s/CMakeFiles/hls_v4l2s.dir/src/hlsV4l2Capture.cpp.o.requires:

.PHONY : HLS_V4l2s/CMakeFiles/hls_v4l2s.dir/src/hlsV4l2Capture.cpp.o.requires

HLS_V4l2s/CMakeFiles/hls_v4l2s.dir/src/hlsV4l2Capture.cpp.o.provides: HLS_V4l2s/CMakeFiles/hls_v4l2s.dir/src/hlsV4l2Capture.cpp.o.requires
	$(MAKE) -f HLS_V4l2s/CMakeFiles/hls_v4l2s.dir/build.make HLS_V4l2s/CMakeFiles/hls_v4l2s.dir/src/hlsV4l2Capture.cpp.o.provides.build
.PHONY : HLS_V4l2s/CMakeFiles/hls_v4l2s.dir/src/hlsV4l2Capture.cpp.o.provides

HLS_V4l2s/CMakeFiles/hls_v4l2s.dir/src/hlsV4l2Capture.cpp.o.provides.build: HLS_V4l2s/CMakeFiles/hls_v4l2s.dir/src/hlsV4l2Capture.cpp.o


HLS_V4l2s/CMakeFiles/hls_v4l2s.dir/src/hlsV4l2Device.cpp.o: HLS_V4l2s/CMakeFiles/hls_v4l2s.dir/flags.make
HLS_V4l2s/CMakeFiles/hls_v4l2s.dir/src/hlsV4l2Device.cpp.o: ../HLS_V4l2s/src/hlsV4l2Device.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/workspace/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object HLS_V4l2s/CMakeFiles/hls_v4l2s.dir/src/hlsV4l2Device.cpp.o"
	cd /workspace/build/HLS_V4l2s && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/hls_v4l2s.dir/src/hlsV4l2Device.cpp.o -c /workspace/HLS_V4l2s/src/hlsV4l2Device.cpp

HLS_V4l2s/CMakeFiles/hls_v4l2s.dir/src/hlsV4l2Device.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/hls_v4l2s.dir/src/hlsV4l2Device.cpp.i"
	cd /workspace/build/HLS_V4l2s && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /workspace/HLS_V4l2s/src/hlsV4l2Device.cpp > CMakeFiles/hls_v4l2s.dir/src/hlsV4l2Device.cpp.i

HLS_V4l2s/CMakeFiles/hls_v4l2s.dir/src/hlsV4l2Device.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/hls_v4l2s.dir/src/hlsV4l2Device.cpp.s"
	cd /workspace/build/HLS_V4l2s && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /workspace/HLS_V4l2s/src/hlsV4l2Device.cpp -o CMakeFiles/hls_v4l2s.dir/src/hlsV4l2Device.cpp.s

HLS_V4l2s/CMakeFiles/hls_v4l2s.dir/src/hlsV4l2Device.cpp.o.requires:

.PHONY : HLS_V4l2s/CMakeFiles/hls_v4l2s.dir/src/hlsV4l2Device.cpp.o.requires

HLS_V4l2s/CMakeFiles/hls_v4l2s.dir/src/hlsV4l2Device.cpp.o.provides: HLS_V4l2s/CMakeFiles/hls_v4l2s.dir/src/hlsV4l2Device.cpp.o.requires
	$(MAKE) -f HLS_V4l2s/CMakeFiles/hls_v4l2s.dir/build.make HLS_V4l2s/CMakeFiles/hls_v4l2s.dir/src/hlsV4l2Device.cpp.o.provides.build
.PHONY : HLS_V4l2s/CMakeFiles/hls_v4l2s.dir/src/hlsV4l2Device.cpp.o.provides

HLS_V4l2s/CMakeFiles/hls_v4l2s.dir/src/hlsV4l2Device.cpp.o.provides.build: HLS_V4l2s/CMakeFiles/hls_v4l2s.dir/src/hlsV4l2Device.cpp.o


HLS_V4l2s/CMakeFiles/hls_v4l2s.dir/src/hlsV4l2MmapDevice.cpp.o: HLS_V4l2s/CMakeFiles/hls_v4l2s.dir/flags.make
HLS_V4l2s/CMakeFiles/hls_v4l2s.dir/src/hlsV4l2MmapDevice.cpp.o: ../HLS_V4l2s/src/hlsV4l2MmapDevice.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/workspace/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CXX object HLS_V4l2s/CMakeFiles/hls_v4l2s.dir/src/hlsV4l2MmapDevice.cpp.o"
	cd /workspace/build/HLS_V4l2s && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/hls_v4l2s.dir/src/hlsV4l2MmapDevice.cpp.o -c /workspace/HLS_V4l2s/src/hlsV4l2MmapDevice.cpp

HLS_V4l2s/CMakeFiles/hls_v4l2s.dir/src/hlsV4l2MmapDevice.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/hls_v4l2s.dir/src/hlsV4l2MmapDevice.cpp.i"
	cd /workspace/build/HLS_V4l2s && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /workspace/HLS_V4l2s/src/hlsV4l2MmapDevice.cpp > CMakeFiles/hls_v4l2s.dir/src/hlsV4l2MmapDevice.cpp.i

HLS_V4l2s/CMakeFiles/hls_v4l2s.dir/src/hlsV4l2MmapDevice.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/hls_v4l2s.dir/src/hlsV4l2MmapDevice.cpp.s"
	cd /workspace/build/HLS_V4l2s && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /workspace/HLS_V4l2s/src/hlsV4l2MmapDevice.cpp -o CMakeFiles/hls_v4l2s.dir/src/hlsV4l2MmapDevice.cpp.s

HLS_V4l2s/CMakeFiles/hls_v4l2s.dir/src/hlsV4l2MmapDevice.cpp.o.requires:

.PHONY : HLS_V4l2s/CMakeFiles/hls_v4l2s.dir/src/hlsV4l2MmapDevice.cpp.o.requires

HLS_V4l2s/CMakeFiles/hls_v4l2s.dir/src/hlsV4l2MmapDevice.cpp.o.provides: HLS_V4l2s/CMakeFiles/hls_v4l2s.dir/src/hlsV4l2MmapDevice.cpp.o.requires
	$(MAKE) -f HLS_V4l2s/CMakeFiles/hls_v4l2s.dir/build.make HLS_V4l2s/CMakeFiles/hls_v4l2s.dir/src/hlsV4l2MmapDevice.cpp.o.provides.build
.PHONY : HLS_V4l2s/CMakeFiles/hls_v4l2s.dir/src/hlsV4l2MmapDevice.cpp.o.provides

HLS_V4l2s/CMakeFiles/hls_v4l2s.dir/src/hlsV4l2MmapDevice.cpp.o.provides.build: HLS_V4l2s/CMakeFiles/hls_v4l2s.dir/src/hlsV4l2MmapDevice.cpp.o


HLS_V4l2s/CMakeFiles/hls_v4l2s.dir/src/hlsV4l2Output.cpp.o: HLS_V4l2s/CMakeFiles/hls_v4l2s.dir/flags.make
HLS_V4l2s/CMakeFiles/hls_v4l2s.dir/src/hlsV4l2Output.cpp.o: ../HLS_V4l2s/src/hlsV4l2Output.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/workspace/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Building CXX object HLS_V4l2s/CMakeFiles/hls_v4l2s.dir/src/hlsV4l2Output.cpp.o"
	cd /workspace/build/HLS_V4l2s && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/hls_v4l2s.dir/src/hlsV4l2Output.cpp.o -c /workspace/HLS_V4l2s/src/hlsV4l2Output.cpp

HLS_V4l2s/CMakeFiles/hls_v4l2s.dir/src/hlsV4l2Output.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/hls_v4l2s.dir/src/hlsV4l2Output.cpp.i"
	cd /workspace/build/HLS_V4l2s && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /workspace/HLS_V4l2s/src/hlsV4l2Output.cpp > CMakeFiles/hls_v4l2s.dir/src/hlsV4l2Output.cpp.i

HLS_V4l2s/CMakeFiles/hls_v4l2s.dir/src/hlsV4l2Output.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/hls_v4l2s.dir/src/hlsV4l2Output.cpp.s"
	cd /workspace/build/HLS_V4l2s && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /workspace/HLS_V4l2s/src/hlsV4l2Output.cpp -o CMakeFiles/hls_v4l2s.dir/src/hlsV4l2Output.cpp.s

HLS_V4l2s/CMakeFiles/hls_v4l2s.dir/src/hlsV4l2Output.cpp.o.requires:

.PHONY : HLS_V4l2s/CMakeFiles/hls_v4l2s.dir/src/hlsV4l2Output.cpp.o.requires

HLS_V4l2s/CMakeFiles/hls_v4l2s.dir/src/hlsV4l2Output.cpp.o.provides: HLS_V4l2s/CMakeFiles/hls_v4l2s.dir/src/hlsV4l2Output.cpp.o.requires
	$(MAKE) -f HLS_V4l2s/CMakeFiles/hls_v4l2s.dir/build.make HLS_V4l2s/CMakeFiles/hls_v4l2s.dir/src/hlsV4l2Output.cpp.o.provides.build
.PHONY : HLS_V4l2s/CMakeFiles/hls_v4l2s.dir/src/hlsV4l2Output.cpp.o.provides

HLS_V4l2s/CMakeFiles/hls_v4l2s.dir/src/hlsV4l2Output.cpp.o.provides.build: HLS_V4l2s/CMakeFiles/hls_v4l2s.dir/src/hlsV4l2Output.cpp.o


HLS_V4l2s/CMakeFiles/hls_v4l2s.dir/src/hlsV4l2ReadWriteDevice.cpp.o: HLS_V4l2s/CMakeFiles/hls_v4l2s.dir/flags.make
HLS_V4l2s/CMakeFiles/hls_v4l2s.dir/src/hlsV4l2ReadWriteDevice.cpp.o: ../HLS_V4l2s/src/hlsV4l2ReadWriteDevice.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/workspace/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "Building CXX object HLS_V4l2s/CMakeFiles/hls_v4l2s.dir/src/hlsV4l2ReadWriteDevice.cpp.o"
	cd /workspace/build/HLS_V4l2s && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/hls_v4l2s.dir/src/hlsV4l2ReadWriteDevice.cpp.o -c /workspace/HLS_V4l2s/src/hlsV4l2ReadWriteDevice.cpp

HLS_V4l2s/CMakeFiles/hls_v4l2s.dir/src/hlsV4l2ReadWriteDevice.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/hls_v4l2s.dir/src/hlsV4l2ReadWriteDevice.cpp.i"
	cd /workspace/build/HLS_V4l2s && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /workspace/HLS_V4l2s/src/hlsV4l2ReadWriteDevice.cpp > CMakeFiles/hls_v4l2s.dir/src/hlsV4l2ReadWriteDevice.cpp.i

HLS_V4l2s/CMakeFiles/hls_v4l2s.dir/src/hlsV4l2ReadWriteDevice.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/hls_v4l2s.dir/src/hlsV4l2ReadWriteDevice.cpp.s"
	cd /workspace/build/HLS_V4l2s && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /workspace/HLS_V4l2s/src/hlsV4l2ReadWriteDevice.cpp -o CMakeFiles/hls_v4l2s.dir/src/hlsV4l2ReadWriteDevice.cpp.s

HLS_V4l2s/CMakeFiles/hls_v4l2s.dir/src/hlsV4l2ReadWriteDevice.cpp.o.requires:

.PHONY : HLS_V4l2s/CMakeFiles/hls_v4l2s.dir/src/hlsV4l2ReadWriteDevice.cpp.o.requires

HLS_V4l2s/CMakeFiles/hls_v4l2s.dir/src/hlsV4l2ReadWriteDevice.cpp.o.provides: HLS_V4l2s/CMakeFiles/hls_v4l2s.dir/src/hlsV4l2ReadWriteDevice.cpp.o.requires
	$(MAKE) -f HLS_V4l2s/CMakeFiles/hls_v4l2s.dir/build.make HLS_V4l2s/CMakeFiles/hls_v4l2s.dir/src/hlsV4l2ReadWriteDevice.cpp.o.provides.build
.PHONY : HLS_V4l2s/CMakeFiles/hls_v4l2s.dir/src/hlsV4l2ReadWriteDevice.cpp.o.provides

HLS_V4l2s/CMakeFiles/hls_v4l2s.dir/src/hlsV4l2ReadWriteDevice.cpp.o.provides.build: HLS_V4l2s/CMakeFiles/hls_v4l2s.dir/src/hlsV4l2ReadWriteDevice.cpp.o


HLS_V4l2s/CMakeFiles/hls_v4l2s.dir/src/xcl2.cpp.o: HLS_V4l2s/CMakeFiles/hls_v4l2s.dir/flags.make
HLS_V4l2s/CMakeFiles/hls_v4l2s.dir/src/xcl2.cpp.o: ../HLS_V4l2s/src/xcl2.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/workspace/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_7) "Building CXX object HLS_V4l2s/CMakeFiles/hls_v4l2s.dir/src/xcl2.cpp.o"
	cd /workspace/build/HLS_V4l2s && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/hls_v4l2s.dir/src/xcl2.cpp.o -c /workspace/HLS_V4l2s/src/xcl2.cpp

HLS_V4l2s/CMakeFiles/hls_v4l2s.dir/src/xcl2.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/hls_v4l2s.dir/src/xcl2.cpp.i"
	cd /workspace/build/HLS_V4l2s && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /workspace/HLS_V4l2s/src/xcl2.cpp > CMakeFiles/hls_v4l2s.dir/src/xcl2.cpp.i

HLS_V4l2s/CMakeFiles/hls_v4l2s.dir/src/xcl2.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/hls_v4l2s.dir/src/xcl2.cpp.s"
	cd /workspace/build/HLS_V4l2s && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /workspace/HLS_V4l2s/src/xcl2.cpp -o CMakeFiles/hls_v4l2s.dir/src/xcl2.cpp.s

HLS_V4l2s/CMakeFiles/hls_v4l2s.dir/src/xcl2.cpp.o.requires:

.PHONY : HLS_V4l2s/CMakeFiles/hls_v4l2s.dir/src/xcl2.cpp.o.requires

HLS_V4l2s/CMakeFiles/hls_v4l2s.dir/src/xcl2.cpp.o.provides: HLS_V4l2s/CMakeFiles/hls_v4l2s.dir/src/xcl2.cpp.o.requires
	$(MAKE) -f HLS_V4l2s/CMakeFiles/hls_v4l2s.dir/build.make HLS_V4l2s/CMakeFiles/hls_v4l2s.dir/src/xcl2.cpp.o.provides.build
.PHONY : HLS_V4l2s/CMakeFiles/hls_v4l2s.dir/src/xcl2.cpp.o.provides

HLS_V4l2s/CMakeFiles/hls_v4l2s.dir/src/xcl2.cpp.o.provides.build: HLS_V4l2s/CMakeFiles/hls_v4l2s.dir/src/xcl2.cpp.o


# Object files for target hls_v4l2s
hls_v4l2s_OBJECTS = \
"CMakeFiles/hls_v4l2s.dir/src/hlsV4l2Access.cpp.o" \
"CMakeFiles/hls_v4l2s.dir/src/hlsV4l2Capture.cpp.o" \
"CMakeFiles/hls_v4l2s.dir/src/hlsV4l2Device.cpp.o" \
"CMakeFiles/hls_v4l2s.dir/src/hlsV4l2MmapDevice.cpp.o" \
"CMakeFiles/hls_v4l2s.dir/src/hlsV4l2Output.cpp.o" \
"CMakeFiles/hls_v4l2s.dir/src/hlsV4l2ReadWriteDevice.cpp.o" \
"CMakeFiles/hls_v4l2s.dir/src/xcl2.cpp.o"

# External object files for target hls_v4l2s
hls_v4l2s_EXTERNAL_OBJECTS =

HLS_V4l2s/libhls_v4l2s.so: HLS_V4l2s/CMakeFiles/hls_v4l2s.dir/src/hlsV4l2Access.cpp.o
HLS_V4l2s/libhls_v4l2s.so: HLS_V4l2s/CMakeFiles/hls_v4l2s.dir/src/hlsV4l2Capture.cpp.o
HLS_V4l2s/libhls_v4l2s.so: HLS_V4l2s/CMakeFiles/hls_v4l2s.dir/src/hlsV4l2Device.cpp.o
HLS_V4l2s/libhls_v4l2s.so: HLS_V4l2s/CMakeFiles/hls_v4l2s.dir/src/hlsV4l2MmapDevice.cpp.o
HLS_V4l2s/libhls_v4l2s.so: HLS_V4l2s/CMakeFiles/hls_v4l2s.dir/src/hlsV4l2Output.cpp.o
HLS_V4l2s/libhls_v4l2s.so: HLS_V4l2s/CMakeFiles/hls_v4l2s.dir/src/hlsV4l2ReadWriteDevice.cpp.o
HLS_V4l2s/libhls_v4l2s.so: HLS_V4l2s/CMakeFiles/hls_v4l2s.dir/src/xcl2.cpp.o
HLS_V4l2s/libhls_v4l2s.so: HLS_V4l2s/CMakeFiles/hls_v4l2s.dir/build.make
HLS_V4l2s/libhls_v4l2s.so: /usr/local/lib/libopencv_video.so.3.4.0
HLS_V4l2s/libhls_v4l2s.so: /usr/local/lib/libopencv_highgui.so.3.4.0
HLS_V4l2s/libhls_v4l2s.so: /usr/local/lib/libopencv_videoio.so.3.4.0
HLS_V4l2s/libhls_v4l2s.so: /usr/local/lib/libopencv_imgcodecs.so.3.4.0
HLS_V4l2s/libhls_v4l2s.so: /usr/local/lib/libopencv_imgproc.so.3.4.0
HLS_V4l2s/libhls_v4l2s.so: /usr/local/lib/libopencv_core.so.3.4.0
HLS_V4l2s/libhls_v4l2s.so: HLS_V4l2s/CMakeFiles/hls_v4l2s.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/workspace/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_8) "Linking CXX shared library libhls_v4l2s.so"
	cd /workspace/build/HLS_V4l2s && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/hls_v4l2s.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
HLS_V4l2s/CMakeFiles/hls_v4l2s.dir/build: HLS_V4l2s/libhls_v4l2s.so

.PHONY : HLS_V4l2s/CMakeFiles/hls_v4l2s.dir/build

HLS_V4l2s/CMakeFiles/hls_v4l2s.dir/requires: HLS_V4l2s/CMakeFiles/hls_v4l2s.dir/src/hlsV4l2Access.cpp.o.requires
HLS_V4l2s/CMakeFiles/hls_v4l2s.dir/requires: HLS_V4l2s/CMakeFiles/hls_v4l2s.dir/src/hlsV4l2Capture.cpp.o.requires
HLS_V4l2s/CMakeFiles/hls_v4l2s.dir/requires: HLS_V4l2s/CMakeFiles/hls_v4l2s.dir/src/hlsV4l2Device.cpp.o.requires
HLS_V4l2s/CMakeFiles/hls_v4l2s.dir/requires: HLS_V4l2s/CMakeFiles/hls_v4l2s.dir/src/hlsV4l2MmapDevice.cpp.o.requires
HLS_V4l2s/CMakeFiles/hls_v4l2s.dir/requires: HLS_V4l2s/CMakeFiles/hls_v4l2s.dir/src/hlsV4l2Output.cpp.o.requires
HLS_V4l2s/CMakeFiles/hls_v4l2s.dir/requires: HLS_V4l2s/CMakeFiles/hls_v4l2s.dir/src/hlsV4l2ReadWriteDevice.cpp.o.requires
HLS_V4l2s/CMakeFiles/hls_v4l2s.dir/requires: HLS_V4l2s/CMakeFiles/hls_v4l2s.dir/src/xcl2.cpp.o.requires

.PHONY : HLS_V4l2s/CMakeFiles/hls_v4l2s.dir/requires

HLS_V4l2s/CMakeFiles/hls_v4l2s.dir/clean:
	cd /workspace/build/HLS_V4l2s && $(CMAKE_COMMAND) -P CMakeFiles/hls_v4l2s.dir/cmake_clean.cmake
.PHONY : HLS_V4l2s/CMakeFiles/hls_v4l2s.dir/clean

HLS_V4l2s/CMakeFiles/hls_v4l2s.dir/depend:
	cd /workspace/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /workspace /workspace/HLS_V4l2s /workspace/build /workspace/build/HLS_V4l2s /workspace/build/HLS_V4l2s/CMakeFiles/hls_v4l2s.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : HLS_V4l2s/CMakeFiles/hls_v4l2s.dir/depend

