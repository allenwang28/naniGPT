"""DeviceMesh creation — no globals.

Created once in the training loop, passed explicitly everywhere.
Adding a new parallelism dimension means adding a mesh dimension here.
"""

import logging

from torch.distributed.device_mesh import DeviceMesh

log = logging.getLogger("distributed.mesh")


def create_device_mesh(world_size: int) -> DeviceMesh:
    """Create a DeviceMesh from the world size.

    Currently 1D with dim name "dp". Extends naturally to multi-dimensional
    meshes like ("pp", "dp", "ep", "tp") when those parallelisms are added —
    TP innermost (NVLink), PP outermost (cross-node).
    """
    return DeviceMesh("cuda", list(range(world_size)), mesh_dim_names=("dp",))
