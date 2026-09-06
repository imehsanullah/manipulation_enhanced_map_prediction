import ast, time, math
from pathlib import Path
from types import SimpleNamespace as NS
import numpy as np
import torch
root=Path(__file__).resolve().parents[1]
def method(arm, name, extra=None, file='scripts/run_cnabu_pipeline.py'):
    tree=ast.parse((root/'shelf_gym'/file).read_text())
    node=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name==name)
    ns=dict(np=np,torch=torch,time=time,math=math)
    ns.update(extra or {})
    exec(compile(ast.Module(body=[node],type_ignores=[]),'<actual-method>','exec'),ns)
    return ns[name]
def probe(arm, allow_push=True):
    seen=[]; count=[0]
    def candidates(*a,**k):
        count[0]+=1
        return {'paths':[1] if count[0]==1 and allow_push else None,'path_annotations':[None],
            'possible_previous_maps':[np.array(0)],'possible_semantic_maps':[np.array(0)]}
    def observe(views,view,b,s):
        seen.append(view); views.append(view); return b,s
    def store(data,b,s,gt,*a,**k): return data,gt
    env=NS(current_obj_ids=[1],n_classes=15,action_budget=8,max_sampled_pushes=80,prob_cutoff=.85,stopping_criterion=.99,ig_calc=None,
       obj=NS(update_obj_states=lambda ids:([np.zeros(3)],None),check_all_object_drop=lambda ids:False),
       get_processed_array_and_gt_data=lambda:({'height_maps':np.ones((1,1,1,1)),'semantic_maps':np.zeros((1,1,1))},{}),
       map_completion_model=NS(dp=NS(get_initial_map=lambda x:(0,0))),get_possible_maps_push=candidates,
       eval_push_igs=lambda *a,**k:(7,0,1000),execute_observation=observe,get_semantic_certainty=lambda x:None,
       get_certainly_mapped_fraction=lambda *a:.5,store_results=store)
    run=method(arm,'run',{'torch':NS(cuda=NS(empty_cache=lambda:None),ones=lambda *a,**k:None),
       'get_igs_for_map':lambda *a,**k:(np.arange(1.,42.),None),
       'get_subsequent_igs_for_map':lambda *a,**k:np.arange(1.,42.),'execute_push':lambda *a,**k:None})
    run(env,debug=False)
    return seen

import unittest
from unittest.mock import MagicMock
from shapely.errors import GEOSException
class ReproductionTests(unittest.TestCase):
    def test_push_selected_view_is_used(self):
        seen=probe('view')
        self.assertEqual(seen[3],7)
        self.assertEqual(len(seen),7)

    def test_no_push_executes_every_observation(self):
        self.assertEqual(len(probe('view',False)),8)

    def test_geometry_guard_preserves_valid_candidates(self):
        ns=dict(GEOSException=GEOSException,object_to_push=lambda *a:np.ones((3,3)),compute_center=lambda *a:(1,1),generate_directions=lambda *a,**k:[],find_best_push_direction=lambda *a:[(([1],2,3,(1,1),(2,2),.5,.5),1)])
        get=method('view','get_samples',ns,file='utils/uncertainty_informed_push_utils.py')
        args=([0,0,0,(1,1)],(2,2),None,None,None)
        def bad(*a):raise GEOSException('Edge direction cannot be determined because endpoints are equal')
        get.__globals__['push_sample']=bad
        self.assertIsNone(get(*args))
        get.__globals__['push_sample']=lambda *a:np.array([[[1,2],[3,4]]])
        np.testing.assert_array_equal(get(*args),np.array([[[22,17],[24,19]]]))
        def unrelated(*a):raise RuntimeError('unrelated')
        get.__globals__['push_sample']=unrelated
        with self.assertRaises(RuntimeError):get(*args)

    def test_optional_seed(self):
        tree=ast.parse((root/'tools/reproduction/run_mem.py').read_text())
        node=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='seed')
        ns=dict(args=NS(seed=None),random=MagicMock(),np=MagicMock(),torch=MagicMock(),cp=MagicMock())
        exec(compile(ast.Module(body=[node],type_ignores=[]),'<seed>','exec'),ns)
        env=NS(rng='original',ps=NS(rng='original'))
        ns['seed'](env)
        self.assertEqual(env.rng,'original')
        for name in ['random','np','torch','cp']:self.assertEqual(ns[name].mock_calls,[])
        ns['args'].seed=23
        ns['seed'](env)
        ns['random'].seed.assert_called_once_with(23)
        ns['torch'].manual_seed.assert_called_once_with(23)
        self.assertEqual(ns['np'].random.default_rng.call_count,2)

if __name__=='__main__':unittest.main()
