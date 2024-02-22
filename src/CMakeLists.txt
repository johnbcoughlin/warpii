add_library(warpii
    INTERFACE
    rk.h
    function_eval.h
    )

#set_target_properties(warpii PROPERTIES LINKER_LANGUAGE CXX)

target_include_directories(warpii INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
#deal_ii_setup_target(warpii)
