# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.22

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
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
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/HwHiAiUser/guodong/workspace/ascend-s2/Lerp/Lerp

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/HwHiAiUser/guodong/workspace/ascend-s2/Lerp/Lerp/build_out

# Utility rule file for ascendc_bin_ascend310b_lerp_1.

# Include any custom commands dependencies for this target.
include op_kernel/CMakeFiles/ascendc_bin_ascend310b_lerp_1.dir/compiler_depend.make

# Include the progress variables for this target.
include op_kernel/CMakeFiles/ascendc_bin_ascend310b_lerp_1.dir/progress.make

op_kernel/CMakeFiles/ascendc_bin_ascend310b_lerp_1:
	cd /home/HwHiAiUser/guodong/workspace/ascend-s2/Lerp/Lerp/build_out/op_kernel/binary/ascend310b && export HI_PYTHON=python3 && bash /home/HwHiAiUser/guodong/workspace/ascend-s2/Lerp/Lerp/build_out/op_kernel/binary/ascend310b/gen/Lerp-lerp-1.sh /home/HwHiAiUser/guodong/workspace/ascend-s2/Lerp/Lerp/build_out/op_kernel/binary/ascend310b/src/Lerp.py /home/HwHiAiUser/guodong/workspace/ascend-s2/Lerp/Lerp/build_out/op_kernel/binary/ascend310b/bin/lerp && echo $(MAKE)

ascendc_bin_ascend310b_lerp_1: op_kernel/CMakeFiles/ascendc_bin_ascend310b_lerp_1
ascendc_bin_ascend310b_lerp_1: op_kernel/CMakeFiles/ascendc_bin_ascend310b_lerp_1.dir/build.make
.PHONY : ascendc_bin_ascend310b_lerp_1

# Rule to build all files generated by this target.
op_kernel/CMakeFiles/ascendc_bin_ascend310b_lerp_1.dir/build: ascendc_bin_ascend310b_lerp_1
.PHONY : op_kernel/CMakeFiles/ascendc_bin_ascend310b_lerp_1.dir/build

op_kernel/CMakeFiles/ascendc_bin_ascend310b_lerp_1.dir/clean:
	cd /home/HwHiAiUser/guodong/workspace/ascend-s2/Lerp/Lerp/build_out/op_kernel && $(CMAKE_COMMAND) -P CMakeFiles/ascendc_bin_ascend310b_lerp_1.dir/cmake_clean.cmake
.PHONY : op_kernel/CMakeFiles/ascendc_bin_ascend310b_lerp_1.dir/clean

op_kernel/CMakeFiles/ascendc_bin_ascend310b_lerp_1.dir/depend:
	cd /home/HwHiAiUser/guodong/workspace/ascend-s2/Lerp/Lerp/build_out && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/HwHiAiUser/guodong/workspace/ascend-s2/Lerp/Lerp /home/HwHiAiUser/guodong/workspace/ascend-s2/Lerp/Lerp/op_kernel /home/HwHiAiUser/guodong/workspace/ascend-s2/Lerp/Lerp/build_out /home/HwHiAiUser/guodong/workspace/ascend-s2/Lerp/Lerp/build_out/op_kernel /home/HwHiAiUser/guodong/workspace/ascend-s2/Lerp/Lerp/build_out/op_kernel/CMakeFiles/ascendc_bin_ascend310b_lerp_1.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : op_kernel/CMakeFiles/ascendc_bin_ascend310b_lerp_1.dir/depend

