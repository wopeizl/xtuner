from unittest.mock import patch

import pytest

from xtuner.v1.module.dispatcher import build_dispatcher


class _ExplodingProcessGroup:
    def size(self):
        pytest.fail("DeepMoE guard inspected the process group before rejecting")


@pytest.mark.parametrize("backend", ["deepmoe", "moonep", "ultraep"])
def test_build_dispatcher_rejects_model_owned_deepmoe_before_group_inspection(backend):
    with patch("xtuner.v1.module.dispatcher.NaiveDispatcher") as naive_dispatcher:
        with pytest.raises(ValueError, match="model-owned.*DeepMoE"):
            build_dispatcher(
                dispatcher=backend,
                n_routed_experts=8,
                ep_group=_ExplodingProcessGroup(),
            )

    naive_dispatcher.assert_not_called()


@pytest.mark.parametrize("backend", ["deepmoe", "moonep", "ultraep"])
def test_build_dispatcher_never_falls_back_when_deepmoe_group_is_none(backend):
    with patch("xtuner.v1.module.dispatcher.NaiveDispatcher") as naive_dispatcher:
        with pytest.raises(ValueError, match=f"dispatcher='{backend}'"):
            build_dispatcher(dispatcher=backend, n_routed_experts=8, ep_group=None)

    naive_dispatcher.assert_not_called()
