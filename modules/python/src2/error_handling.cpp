#include "error_handling.hpp"

#include "opencv2/core/utils/configuration.private.hpp"
#include "opencv2/core/utils/logger.hpp"

#include "pycompat.hpp"

namespace pycv {
namespace detail {
static bool isPythonBindingsDebugEnabled()
{
    static bool param_debug = cv::utils::getConfigurationParameterBool("OPENCV_PYTHON_DEBUG", false);
    return param_debug;
}

void raisePyError(PyObject* exc, const char* message)
{
    static bool param_debug = isPythonBindingsDebugEnabled();
    if (param_debug)
    {
        CV_LOG_WARNING(CV_NULL_PTR, "Bindings conversion failed: " << message);
    }
    PyErr_SetString(exc, message);
}

int onError(int status, const char* func_name, const char* err_msg, const char* file_name,
            int line, void* userdata)
{
    pycv::PyEnsureGIL gil;

    PyObject* on_error = (PyObject*)userdata;
    PyObject* args = Py_BuildValue("isssi", status, func_name, err_msg, file_name, line);

    PyObject* r = PyObject_Call(on_error, args, CV_NULL_PTR);
    if (!r)
    {
        PyErr_Print();
    }
    else
    {
        Py_DECREF(r);
    }
    Py_DECREF(args);
    return 0; // The return value isn't used
}
} // namespace detail

#ifdef CV_CXX11
template<>
void raisePyError(PyObject* errorType, const char* fmt)
{
    detail::raisePyError(errorType, fmt);
}
#else
void raisePyError(PyObject* errorType, const char* fmt, va_list args)
{
    char message[1000];
    vsnprintf(message, sizeof(message), fmt, args);
    detail::raisePyError(errorType, message);
}

void raiseTypeError(const char* fmt, ...)
{
    va_list args;
    va_start(args, fmt);
    raisePyError(PyExc_TypeError, fmt, args);
    va_end(args);
}

void raiseValueError(const char* fmt, ...)
{
    va_list args;
    va_start(args, fmt);
    raisePyError(PyExc_ValueError, fmt, args);
    va_end(args);
}
#endif


static PyObject* opencv_error = CV_NULL_PTR;


void raiseCVException(const cv::Exception& e)
{
    PyObject_SetAttrString(opencv_error, "file", PyString_FromString(e.file.c_str()));
    PyObject_SetAttrString(opencv_error, "func", PyString_FromString(e.func.c_str()));
    PyObject_SetAttrString(opencv_error, "line", PyInt_FromLong(e.line));
    PyObject_SetAttrString(opencv_error, "code", PyInt_FromLong(e.code));
    PyObject_SetAttrString(opencv_error, "msg", PyString_FromString(e.msg.c_str()));
    PyObject_SetAttrString(opencv_error, "err", PyString_FromString(e.err.c_str()));
    PyErr_SetString(opencv_error, e.what());
}

PyObject* configureException(char* errorName)
{
    PyObject* opencv_error_dict = PyDict_New();
    PyDict_SetItemString(opencv_error_dict, "file", Py_None);
    PyDict_SetItemString(opencv_error_dict, "func", Py_None);
    PyDict_SetItemString(opencv_error_dict, "line", Py_None);
    PyDict_SetItemString(opencv_error_dict, "code", Py_None);
    PyDict_SetItemString(opencv_error_dict, "msg", Py_None);
    PyDict_SetItemString(opencv_error_dict, "err", Py_None);
    opencv_error = PyErr_NewException(errorName, CV_NULL_PTR, opencv_error_dict);
    Py_DECREF(opencv_error_dict);
    return opencv_error;
}

PyObject* redirectError(PyObject*, PyObject* args, PyObject* kw)
{
    const char* keywords[] = { "on_error", CV_NULL_PTR };
    PyObject* on_error = CV_NULL_PTR;

    if (!PyArg_ParseTupleAndKeywords(args, kw, "O", (char**)keywords, &on_error))
    {
        return CV_NULL_PTR;
    }

    if ((on_error != Py_None) && !PyCallable_Check(on_error))
    {
        raiseTypeError("on_error must be callable");
        return CV_NULL_PTR;
    }

    // Keep track of the previous handler parameter, so we can decref it when no longer used
    static PyObject* last_on_error = CV_NULL_PTR;
    if (last_on_error)
    {
        Py_DECREF(last_on_error);
        last_on_error = CV_NULL_PTR;
    }

    if (on_error == Py_None)
    {
        ERRWRAP2(cv::redirectError(CV_NULL_PTR));
    }
    else
    {
        last_on_error = on_error;
        Py_INCREF(last_on_error);
        ERRWRAP2(cv::redirectError(detail::onError, last_on_error));
    }
    Py_RETURN_NONE;
}
} // namespace pycv