#ifndef DMLC_BUILD_CONFIG_H_
#define DMLC_BUILD_CONFIG_H_

/* #undef DMLC_FOPEN_64_PRESENT */

#if !defined(DMLC_FOPEN_64_PRESENT) && DMLC_USE_FOPEN64
  #define fopen64 std::fopen
#endif

/* #undef DMLC_CXXABI_H_PRESENT */
/* #undef DMLC_EXECINFO_H_PRESENT */

#if (defined DMLC_CXXABI_H_PRESENT) && (defined DMLC_EXECINFO_H_PRESENT)
  #ifndef DMLC_LOG_STACK_TRACE
  #define DMLC_LOG_STACK_TRACE 1
  #endif
  #ifndef DMLC_LOG_STACK_TRACE_SIZE
  #define DMLC_LOG_STACK_TRACE_SIZE 10
  #endif
/* #undef DMLC_EXECINFO_H */
#endif

/* #undef DMLC_NANOSLEEP_PRESENT */

#define DMLC_CMAKE_LITTLE_ENDIAN 1

#endif  // DMLC_BUILD_CONFIG_H_
