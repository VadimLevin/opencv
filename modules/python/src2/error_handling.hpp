// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_ERROR_HANDLING_HPP
#define OPENCV_ERROR_HANDLING_HPP

#include <Python.h>

#include "opencv2/core.hpp"

#include "threading.hpp"

#ifndef CV_NULL_PTR
    #ifdef CV_CXX11
        #define CV_NULL_PTR nullptr
    #else
        #define CV_NULL_PTR NULL
    #endif
#endif

#define ERRWRAP2(expr) \
try \
{ \
    pycv::PyAllowThreads allowThreads; \
    expr; \
} \
catch (const cv::Exception &e) \
{ \
    pycv::raiseCVException(e); \
    return 0; \
}

namespace pycv {
namespace detail {
void raisePyError(PyObject* exc, const char* message);

int onError(int status, const char* func_name, const char* err_msg, const char* file_name,
            int line, void* userdata);

} // namespace detail

#ifdef CV_CXX11
template<class... Args>
void raisePyError(PyObject* errorType, const char* fmt, Args&&... args)
{
    char message[1000];
    snprintf(message, sizeof(message), fmt, args...);
    detail::raisePyError(errorType, message);
}

template<>
void raisePyError(PyObject* errorType, const char* fmt);

template<class... Args>
void raiseTypeError(const char* fmt, Args&&... args)
{
    raisePyError(PyExc_TypeError, fmt, std::forward<Args>(args)...);
}

template<class... Args>
void raiseValueError(const char* fmt, Args&&... args)
{
    raisePyError(PyExc_ValueError, fmt, std::forward<Args>(args)...);
}
#else
void raisePyError(PyObject* errorType, const char* fmt, va_list args);

void raiseTypeError(const char* fmt, ...);

void raiseValueError(const char* fmt, ...);
#endif

void raiseCVException(const cv::Exception& e);

PyObject* configureException(char* errorName);

PyObject* redirectError(PyObject*, PyObject* args, PyObject* kw);

} // namespace pycv

#endif //OPENCV_ERROR_HANDLING_HPP
