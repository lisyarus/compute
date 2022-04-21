file(GLOB_RECURSE BLUR_SOURCES "include/*.hpp" "source/*.cpp")

psemek_add_executable(blur ${BLUR_SOURCES})
if (TARGET blur)
	target_include_directories(blur PUBLIC include)
endif()
