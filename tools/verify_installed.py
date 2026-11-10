"""Installed outside-checkout reproduction; no source-tree import path added."""
import importlib
import json
from pathlib import Path
import pkgutil
import subprocess
import sys
import causallab

expected=json.loads(Path(sys.argv[1]).read_text());source=Path(sys.argv[2]).resolve()
origins=[]
for name in ['causallab']+['causallab.'+m.name for m in pkgutil.iter_modules(causallab.__path__)]:
    module=importlib.import_module(name);origin=Path(module.__file__).resolve()
    assert 'site-packages' in origin.parts and source not in origin.parents,(name,origin)
    origins.append(str(origin))
entries=[[sys.executable,'-m','causallab'],[str(Path(sys.executable).parent/'causallab')]]
for entry in entries:
    for command,result in expected.items():
        actual=json.loads(subprocess.check_output(entry+[command]))
        assert actual==result,command
    actual=json.loads(subprocess.check_output(entry+['frontdoor-effect',str(source/'examples/frontdoor_graph.json'),'--treatment','X','--outcome','Y','--mediators','M','--joint',str(source/'examples/frontdoor_joint.json')]))
    assert actual==expected['frontdoor-effect-demo']['effect']
print(json.dumps({'demo_reports':len(expected),'entrypoints':len(entries),'exact_effect_queries':2,'module_origins':origins}))
