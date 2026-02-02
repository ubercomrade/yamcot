# Import the C++ extension functionality
try:
    # Try to import from the compiled C++ extension
    # We use absolute import to get the binary module
    import _core
    run_motali_cpp = _core.run_motali_cpp
except (ImportError, AttributeError) as e:
    _import_error = e
    # Fallback when the compiled extension is not available
    def run_motali_cpp(*args, **kwargs):
        raise ImportError(
            "The C++ extension '_core' is not installed or could not be loaded. "
            "Please ensure the package was built correctly. "
            f"Original error: {_import_error}"
        )