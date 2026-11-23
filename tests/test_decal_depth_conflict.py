import pytest
import graphics.decal_depth_conflict as mod

CASES = [
    ("project_rebuild", (1)),
    ("depth_bias", (1)),
    ("normal_blend", (1)),
    ("dynamic_attach", (1)),
    ("sort_order", (1)),
    ("atlas_batch", (1)),
    ("clip_bounds", (1)),
    ("validation", (1)),
    ("audit", (1)),
    ("fallback", (1)),
    ("slope_stretch", (1)),
    ("metrics", (1)),
    ("backface", (1)),
    ("fade", (1)),
    ("determinism", (1)),
]

@pytest.mark.parametrize("fn_name,args", CASES)
def test_contract(fn_name, args):
    fn = getattr(mod, fn_name)
    assert fn(*args) is not None
