Putting my sparse/abstracted linear solvers into one place.

Includes CPU and GPU version of conjugate gradient, conjugate residual, GMRES, and JFNK.

### Dependencies:

- https://github.com/thenumbernine/lua-ext
- OpenCL-based solvers depend on https://github.com/thenumbernine/lua-opencl

The tests also use:
- https://github.com/thenumbernine/vec-lua
- https://github.com/thenumbernine/lua-matrix
- https://github.com/thenumbernine/lua-gnuplot
