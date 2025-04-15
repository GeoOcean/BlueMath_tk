from copy import deepcopy
from typing import Sequence

import jigsawpy
import numpy as np
import ocsmesh
from jigsawpy import jigsaw_msh_t
from shapely import Polygon
from shapely.geometry import MultiPolygon
from shapely.ops import unary_union

ELEM_2D_TYPES = ["tria3", "quad4", "hexa8"]
MESH_TYPES = {"tria3": "TRIA3_t", "quad4": "QUAD4_t", "hexa8": "HEXA8_t"}


def cleanup_skewed_el_M(
    mesh: jigsaw_msh_t,
    lw_bound_tri: float = 1.0,
    up_bound_tri: float = 175.0,
    lw_bound_quad: float = 10.0,
    up_bound_quad: float = 179.0,
) -> jigsaw_msh_t:
    tria = None
    quad = None
    ang_tri, ang_quad = ocsmesh.utils.calc_el_angles(mesh)
    if len(ang_tri) > 0:
        ang_chk_tri = np.logical_or(
            ang_tri[0] < lw_bound_tri, ang_tri[0] >= up_bound_tri
        )
        mask_tri = np.where(np.any(ang_chk_tri, axis=1))[0]
        tria = mask_tri

    mesh_clean = clip_elements_by_index_M(mesh, tria=tria, quad=None, inverse=False)

    return mesh_clean


def clip_elements_by_index_M(
    msht: jigsaw_msh_t, tria=None, quad=None, inverse: bool = False
) -> jigsaw_msh_t:
    new_msht = deepcopy(msht)

    rm_dict = {"tria3": tria, "quad4": quad}
    for elm_type, idx in rm_dict.items():
        if idx is None:
            continue
        mask = np.ones(getattr(new_msht, elm_type).shape, dtype=bool)
        mask[idx] = False
        if inverse is False:
            if elm_type == "tria3":
                setattr(
                    new_msht.msh_t,
                    elm_type,
                    getattr(new_msht, elm_type).take(np.where(mask)[0], axis=0),
                )
            else:
                setattr(
                    new_msht,
                    elm_type,
                    getattr(new_msht, elm_type).take(np.where(mask)[0], axis=0),
                )
        else:
            if elm_type == "tria3":
                setattr(
                    new_msht.msh_t,
                    elm_type,
                    getattr(new_msht, elm_type).take(np.where(~mask)[0], axis=0),
                )
            else:
                setattr(
                    new_msht,
                    elm_type,
                    getattr(new_msht, elm_type).take(np.where(~mask)[0], axis=0),
                )

    return new_msht


def cleanup_isolates_M(mesh):
    used_old_idx = np.array([], dtype="int64")
    for etype in ELEM_2D_TYPES:
        elem_idx = getattr(mesh.msh_t, etype)["index"].flatten()
        used_old_idx = np.hstack((used_old_idx, elem_idx))
    used_old_idx = np.unique(used_old_idx)
    # update vert2 and value
    mesh.msh_t.vert2 = mesh.vert2.take(used_old_idx, axis=0)
    if len(mesh.msh_t.value) > 0:
        mesh.msh_t.value = mesh.msh_t.value.take(used_old_idx, axis=0)

    renum = {old: new for new, old in enumerate(np.unique(used_old_idx))}
    for etype in ELEM_2D_TYPES:
        elem_idx = getattr(mesh.msh_t, etype)["index"]
        elem_new_idx = np.array([renum[i] for i in elem_idx.flatten()])
        elem_new_idx = elem_new_idx.reshape(elem_idx.shape)
        # TODO: Keep IDTag?
        setattr(
            mesh.msh_t,
            etype,
            np.array(
                [(idx, 0) for idx in elem_new_idx],
                dtype=getattr(jigsawpy.jigsaw_msh_t, f"{etype.upper()}_t"),
            ),
        )


def cleanup_pinched_nodes_M(mesh):
    _inner_ring_collection = ocsmesh.utils.inner_ring_collection(mesh)
    all_nodes = []
    for inner_rings in _inner_ring_collection.values():
        for ring in inner_rings:
            all_nodes.extend(np.asarray(ring)[:, 0].tolist())
    u, c = np.unique(all_nodes, return_counts=True)
    mesh.msh_t.tria3 = mesh.tria3.take(
        np.where(~np.any(np.isin(mesh.tria3["index"], u[c > 1]), axis=1))[0], axis=0
    )


def put_id_tags_M(mesh):
    # start enumerating on 1 to avoid issues with indexing on fortran models
    mesh.msh_t.vert2 = np.array(
        [(coord, id + 1) for id, coord in enumerate(mesh.vert2["coord"])],
        dtype=jigsaw_msh_t.VERT2_t,
    )
    mesh.msh_t.tria3 = np.array(
        [(index, id + 1) for id, index in enumerate(mesh.tria3["index"])],
        dtype=jigsaw_msh_t.TRIA3_t,
    )
    mesh.msh_t.quad4 = np.array(
        [(index, id + 1) for id, index in enumerate(mesh.quad4["index"])],
        dtype=jigsaw_msh_t.QUAD4_t,
    )
    mesh.msh_t.hexa8 = np.array(
        [(index, id + 1) for id, index in enumerate(mesh.msh_t.hexa8["index"])],
        dtype=jigsaw_msh_t.HEXA8_t,
    )


def finalize_mesh_M(mesh, sieve_area=None):
    cleanup_isolates_M(mesh)

    while True:
        no_op = True

        pinched_nodes = ocsmesh.utils.get_pinched_nodes(mesh.msh_t)
        if len(pinched_nodes):
            no_op = False
            # TODO drop fewer elements for pinch
            clip_mesh_by_vertex_M(
                mesh,
                pinched_nodes,
                can_use_other_verts=True,
                inverse=True,
                in_place=True,
            )

        boundary_polys = get_mesh_polygons_M(mesh)
        sieve_mask = ocsmesh.utils._get_sieve_mask(mesh, boundary_polys, sieve_area)
        if np.sum(sieve_mask):
            no_op = False
            ocsmesh.utils._sieve_by_mask(mesh, sieve_mask)

        if no_op:
            break

    cleanup_isolates_M(mesh)
    cleanup_duplicates_M(mesh)
    put_id_tags_M(mesh)


def get_mesh_polygons_M(mesh):
    elm_polys = []
    for elm_type in ELEM_2D_TYPES:
        elems = getattr(mesh.msh_t, elm_type)["index"]
        elm_polys.extend([Polygon(mesh.vert2["coord"][cell]) for cell in elems])

    poly = unary_union(elm_polys)
    if isinstance(poly, Polygon):
        poly = MultiPolygon([poly])

    return poly


def cleanup_duplicates_M(mesh):
    """Cleanup duplicate nodes and elements

    Notes
    -----
    Elements and nodes are duplicate if they fully overlapping (not
    partially)
    """

    _, cooidx, coorev = np.unique(
        mesh.vert2["coord"], axis=0, return_index=True, return_inverse=True
    )
    nd_map = dict(enumerate(coorev))
    mesh.msh_t.vert2 = mesh.vert2.take(cooidx, axis=0)

    for etype, otype in MESH_TYPES.items():
        cnn = getattr(mesh.msh_t, etype)["index"]

        n_node = cnn.shape[1]

        cnn_renu = np.array([nd_map[i] for i in cnn.flatten()]).reshape(-1, n_node)

        _, cnnidx = np.unique(np.sort(cnn_renu, axis=1), axis=0, return_index=True)
        mask = np.zeros(len(cnn_renu), dtype=bool)
        mask[cnnidx] = True
        adj_cnn = cnn_renu[mask]

        setattr(
            mesh.msh_t,
            etype,
            np.array([(idx, 0) for idx in adj_cnn], dtype=getattr(jigsaw_msh_t, otype)),
        )

    if len(mesh.value) > 0:
        mesh.msh_t.value = mesh.value.take(sorted(cooidx), axis=0)


def clip_mesh_by_vertex_M(
    mesh: jigsaw_msh_t,
    vert_in: Sequence[int],
    can_use_other_verts: bool = False,
    inverse: bool = False,
    in_place: bool = False,
) -> jigsaw_msh_t:
    if mesh.msh_t.mshID == "euclidean-mesh" and mesh.msh_t.ndims == 2:
        coord = mesh.vert2["coord"]

        # TODO: What about edge2 if in_place?
        elm_dict = {key: getattr(mesh.msh_t, key)["index"] for key in MESH_TYPES}

        # Whether elements that include "in"-vertices can be created
        # using vertices other than "in"-vertices
        mark_func = np.all
        if can_use_other_verts:
            mark_func = np.any

        mark_dict = {
            key: mark_func((np.isin(elems.ravel(), vert_in).reshape(elems.shape)), 1)
            for key, elems in elm_dict.items()
        }

        # Whether to return elements found by "in" vertices or return
        # all elements except them
        if inverse:
            mark_dict = {key: np.logical_not(mark) for key, mark in mark_dict.items()}

        # Find elements based on old vertex index
        elem_draft_dict = {key: elm_dict[key][mark_dict[key], :] for key in elm_dict}

        crd_old_to_new = {
            index: i
            for i, index in enumerate(
                sorted(
                    np.unique(
                        np.concatenate(
                            [draft.ravel() for draft in elem_draft_dict.values()]
                        )
                    )
                )
            )
        }

        elem_final_dict = {
            key: np.array([[crd_old_to_new[x] for x in element] for element in draft])
            for key, draft in elem_draft_dict.items()
        }

        new_coord = coord[list(crd_old_to_new.keys()), :]
        value = np.zeros(shape=(0, 0), dtype=jigsaw_msh_t.REALS_t)
        if len(mesh.value) == len(coord):
            value = mesh.value.take(list(crd_old_to_new.keys()), axis=0).copy()

        mesh_out = mesh
        if not in_place:
            mesh_out = jigsaw_msh_t()
            mesh_out.mshID = mesh.mshID
            mesh_out.ndims = mesh.ndims
            if hasattr(mesh, "crs"):
                mesh_out.crs = deepcopy(mesh.crs)

        mesh_out.msh_t.value = value

        mesh_out.msh_t.vert2 = np.array(
            [(coo, 0) for coo in new_coord], dtype=jigsaw_msh_t.VERT2_t
        )

        for key, elem_type in MESH_TYPES.items():
            setattr(
                mesh_out.msh_t,
                key,
                np.array(
                    [(con, 0) for con in elem_final_dict[key]],
                    dtype=getattr(jigsaw_msh_t, elem_type),
                ),
            )

        return mesh_out

    msg = f"Not implemented for mshID={mesh.mshID} and dim={mesh.ndims}"
    raise NotImplementedError(msg)
