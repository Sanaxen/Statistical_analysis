#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "xgboost::xgboost" for configuration "Release"
set_property(TARGET xgboost::xgboost APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(xgboost::xgboost PROPERTIES
  IMPORTED_IMPLIB_RELEASE "${_IMPORT_PREFIX}/lib/xgboost.lib"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/xgboost.dll"
  )

list(APPEND _IMPORT_CHECK_TARGETS xgboost::xgboost )
list(APPEND _IMPORT_CHECK_FILES_FOR_xgboost::xgboost "${_IMPORT_PREFIX}/lib/xgboost.lib" "${_IMPORT_PREFIX}/bin/xgboost.dll" )

# Import target "xgboost::runxgboost" for configuration "Release"
set_property(TARGET xgboost::runxgboost APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(xgboost::runxgboost PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/xgboost.exe"
  )

list(APPEND _IMPORT_CHECK_TARGETS xgboost::runxgboost )
list(APPEND _IMPORT_CHECK_FILES_FOR_xgboost::runxgboost "${_IMPORT_PREFIX}/bin/xgboost.exe" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
