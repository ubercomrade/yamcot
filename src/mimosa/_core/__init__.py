try:
    from . import _core

    run_motali_cpp = _core.run_motali_cpp
except (ImportError, AttributeError) as e:
    _import_error = e
    _core = None

    def run_motali_cpp(*args, **kwargs):
        """Raise an error when the compiled C++ extension is unavailable."""
        raise ImportError(
            "The C++ extension '_core' is not installed or could not be loaded. "
            "Please ensure the package was built correctly. "
            f"Original error: {_import_error}"
        )
