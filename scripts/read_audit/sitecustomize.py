"""Read audit for the Zenodo deposit: log every file under the data root that a Python process opens for
reading, including HDF5 / netCDF files opened from C (h5py, netCDF4, xarray), and every folder it lists.
Active only when TULEPOND_READ_LOG is set; one line per event: <pid> <kind> <path>."""
import os
import sys

_LOG = os.environ.get("TULEPOND_READ_LOG")
_ROOT = os.environ.get("TULEPOND_AUDIT_ROOT", "").rstrip(os.sep) + os.sep
_busy = [False]                                           # re-entry guard: writing the log must not log itself
if _LOG and _ROOT != os.sep:
    def _record(kind, path):
        if _busy[0]:
            return
        try:
            path = os.path.abspath(os.fsdecode(os.fspath(path)))
        except TypeError:
            return
        if path.startswith(_ROOT):
            _busy[0] = True
            try:
                with open(_LOG, "a", buffering=1) as fh:
                    fh.write(f"{os.getpid()} {kind} {path}\n")
            finally:
                _busy[0] = False

    def _hook(event, args):
        if event == "open" and args and isinstance(args[0], (str, bytes, os.PathLike)):
            mode = args[1] if len(args) > 1 else "r"
            flags = args[2] if len(args) > 2 else 0
            writing = (isinstance(mode, str) and any(c in mode for c in "wax+")) or (isinstance(flags, int) and flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT))
            _record("write" if writing else "read", args[0])
        elif event in ("os.listdir", "os.scandir") and args and args[0] is not None:
            _record("list", args[0])
    sys.addaudithook(_hook)

    import builtins
    _real_import = builtins.__import__

    def _wrap_after_import(name, *a, **k):
        module = _real_import(name, *a, **k)
        try:
            if name == "h5py" and not getattr(module.File, "_audited", False):
                original = module.File.__init__
                def init(self, name, mode="r", *args, **kwargs):
                    _record("read" if mode in (None, "r") else "write", name)
                    return original(self, name, mode, *args, **kwargs)
                module.File.__init__ = init
                module.File._audited = True
            elif name == "netCDF4" and not getattr(module, "_audited", False):
                original_ds = module.Dataset
                class Dataset(original_ds):
                    def __init__(self, filename, mode="r", *args, **kwargs):
                        _record("read" if mode == "r" else "write", filename)
                        super().__init__(filename, mode, *args, **kwargs)
                module.Dataset = Dataset
                module._audited = True
            elif name == "xarray" and not getattr(module, "_audited", False):
                original_open = module.open_dataset
                def open_dataset(filename_or_obj, *args, **kwargs):
                    if isinstance(filename_or_obj, (str, os.PathLike)):
                        _record("read", filename_or_obj)
                    return original_open(filename_or_obj, *args, **kwargs)
                module.open_dataset = open_dataset
                module._audited = True
        except Exception:
            pass
        return module
    builtins.__import__ = _wrap_after_import
