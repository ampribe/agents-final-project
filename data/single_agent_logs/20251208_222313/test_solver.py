import json, importlib.util
spec = importlib.util.spec_from_file_location('solver', 'solver.py')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
example = {
  "proposer_prefs": [[0,1,2],[1,0,2],[0,1,2]],
  "receiver_prefs": [[1,0,2],[0,1,2],[0,1,2]]
}
print(mod.solve(example))
