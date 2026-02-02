# Import the C++ extension functionality
try:
    # Try to import from the compiled C++ extension
    from . import _core
    run_motali_cpp = _core.run_motali_cpp
except ImportError:
    # Fallback when the compiled extension is not available
    run_motali_cpp = None