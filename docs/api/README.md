# Photara library API

Photara exposes native C++ libraries and an optional Python module named
`photara`. Both run reconstruction in the caller's process. They do not invoke
the command-line executable.

| Guide | Contents |
|---|---|
| [Python API](python.md) | Build/import, all bound top-level functions, result handles, callbacks, failure handling, examples |
| [C++ API](cpp.md) | CMake integration, native function contracts, ownership, callbacks, examples and Python correspondence |
| [Types and enumerations](types.md) | Every bound enum, configuration/result field, C++ type, default initializer, source comment and declaration link |

Start with the guide for your language. The shared type reference uses the
Python class name as its heading and gives the corresponding C++ type directly
below it. Python exposes a subset of the native libraries; an entry in a C++
header does not imply a Python binding exists.

## Scope and maintenance

These documents describe the source in this checkout. Read `photara.__version__`
to identify a Python build. There is no independent SDK compatibility or binary
ABI guarantee specified here. Keep headers, libraries, Python extension and
runtime dependencies from a compatible build together.

Function semantics and examples are maintained in the two guides. Field and
enum tables are generated from
[`bindings.cpp`](../../photara/src/python/bindings.cpp) and the public headers:

```powershell
python scripts/generate_api_reference.py
python scripts/generate_api_reference.py --check
```

The generator only needs the Python standard library, not CUDA or a compiled
extension. It resolves direct bound fields against their C++ declarations and
fails on unresolved fields. It copies existing source comments; fields without
comments link to their declaration. When adding a binding, also update the
function guide and run the generator. When extending native APIs beyond the
Python subset, update the C++ guide and the relevant header comments.

Build prerequisites and feature switches are listed in the
[repository README](../../README.md#requirements). Architecture and algorithm
notes remain under [`docs/`](../).
