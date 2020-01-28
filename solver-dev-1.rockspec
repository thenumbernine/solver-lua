package = "solver"
version = "dev-1"
source = {
   url = "git+https://github.com/thenumbernine/solver-lua.git"
}
description = {
   summary = [[
   Putting my sparse/abstracted linear solvers into one place.
]],
   detailed = [[
Putting my sparse/abstracted linear solvers into one place.

Includes CPU and GPU version of conjugate gradient, conjugate residual, gmres, and JFNK.
]],
   homepage = "https://github.com/thenumbernine/solver-lua",
   license = "MIT"
}
dependencies = {
   "lua => 5.1"
}
build = {
   type = "builtin",
   modules = {
      ["solver.backsub"] = "backsub.lua",
      ["solver.bicg"] = "bicg.lua",
      ["solver.bicgstab"] = "bicgstab.lua",
      ["solver.cl.bicgstab"] = "cl/bicgstab.lua",
      ["solver.cl.conjgrad"] = "cl/conjgrad.lua",
      ["solver.cl.conjres"] = "cl/conjres.lua",
      ["solver.cl.gmres"] = "cl/gmres.lua",
      ["solver.cl.jfnk"] = "cl/jfnk.lua",
      ["solver.cl.solver"] = "cl/solver.lua",
      ["solver.conjgrad"] = "conjgrad.lua",
      ["solver.conjres"] = "conjres.lua",
      ["solver.fwdsub"] = "fwdsub.lua",
      ["solver.gmres"] = "gmres.lua",
      ["solver.jacobi"] = "jacobi.lua",
      ["solver.jfnk"] = "jfnk.lua",
      ["solver.lup"] = "lup.lua",
      ["solver.qr_gramschmidt"] = "qr_gramschmidt.lua",
      ["solver.qr_gramschmidt_classic"] = "qr_gramschmidt_classic.lua",
      ["solver.qr_householder"] = "qr_householder.lua",
      ["solver.solve_lup"] = "solve_lup.lua",
      ["solver.solve_qr"] = "solve_qr.lua",
      ["solver.tests.dense"] = "tests/dense.lua",
      ["solver.tests.jfnk-cl"] = "tests/jfnk-cl.lua",
      ["solver.tests.jfnk-gpu"] = "tests/jfnk-gpu.lua",
      ["solver.tests.krylov"] = "tests/krylov.lua",
      ["solver.tests.krylov-cl"] = "tests/krylov-cl.lua",
      ["solver.tests.krylov-cl-separate"] = "tests/krylov-cl-separate.lua",
      ["solver.tests.krylov-cl-subbuffer"] = "tests/krylov-cl-subbuffer.lua"
   }
}
