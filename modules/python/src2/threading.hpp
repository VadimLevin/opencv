// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.


#ifndef OPENCV_THREADING_HPP
#define OPENCV_THREADING_HPP

#include <Python.h>

#ifndef CV_NULL_PTR
    #ifdef CV_CXX11
        #define CV_NULL_PTR nullptr
    #else
        #define CV_NULL_PTR NULL
    #endif
#endif

namespace pycv {
class PyAllowThreads
{
public:
    PyAllowThreads()
        : _state(PyEval_SaveThread())
    {
    }

    ~PyAllowThreads() { PyEval_RestoreThread(_state); }

private:
    PyThreadState* _state { CV_NULL_PTR };
};

class PyEnsureGIL
{
public:
    PyEnsureGIL()
        : _state(PyGILState_Ensure())
    {
    }

    ~PyEnsureGIL() { PyGILState_Release(_state); }

private:
    PyGILState_STATE _state;
};
} // namespace pycv
#endif //OPENCV_THREADING_HPP
