"""ShapeKit-Geodesic thoracolumbar body-core and geodesic refinement stage."""

from __future__ import annotations

import numpy as np
import nibabel as nib
from scipy import ndimage as ndi
from skimage.graph import MCP_Geometric

STRUCTURE = np.ones((3, 3, 3), dtype=bool)
TARGETS = tuple(range(5, 12))


def _largest(mask):
    cc, count = ndi.label(mask, STRUCTURE)
    if not count:
        raise ValueError('required anchor label is empty')
    sizes = np.bincount(cc.ravel()); sizes[0] = 0
    return np.argwhere(cc == sizes.argmax()).mean(axis=0)


def _cores(distance, union, ct, affine, radius, anchor_world):
    cc, _ = ndi.label(distance > radius, STRUCTURE)
    sizes = np.bincount(cc.ravel()); sizes[0] = 0
    vv = abs(np.linalg.det(affine[:3, :3]))
    components = []
    for cid in np.flatnonzero(sizes * vv >= 500):
        pos = np.argwhere(cc == cid)
        world = nib.affines.apply_affine(affine, pos.mean(0))
        # Posterior elements can also have a thick core. Retain anterior body
        # candidates near the left-right course of the anchored column.
        axis_x = np.interp(world[2], anchor_world[:, 2], anchor_world[:, 0])
        axis_y = np.interp(world[2], anchor_world[:, 2], anchor_world[:, 1])
        if abs(world[0] - axis_x) > 25 or world[1] < axis_y - 8:
            continue
        values = ct[tuple(pos.T)]
        if np.mean(values >= 130) < .50:
            continue
        counts = np.bincount(union[tuple(pos.T)], minlength=25)
        components.append(dict(cid=int(cid), center=world,
                               voxels=int(sizes[cid]), counts=counts))
    components.sort(key=lambda c: c['center'][2])
    return cc, components


def _chain(components):
    # Anchor evidence comes from the labels the user identified as reliable;
    # intermediate IDs are intentionally ignored when counting the bodies.
    for low, first in enumerate(components):
        if first['counts'][4] / first['voxels'] < .75:
            continue
        for high in range(low + 1, len(components)):
            last = components[high]
            if last['counts'][12] / last['voxels'] < .75 or high-low != 8:
                continue
            selected = components[low:high+1]
            centers = np.array([c['center'] for c in selected])
            gaps = np.diff(centers[:, 2])
            jumps = np.linalg.norm(np.diff(centers[:, :2], axis=0), axis=1)
            if np.all((gaps >= 12) & (gaps <= 50)) and np.all(jumps <= 25):
                if np.max(np.maximum(gaps[1:]/gaps[:-1], gaps[:-1]/gaps[1:])) <= 1.8:
                    return low, high, selected
    return None


def _posterior_markers(raw, image, affine, centers):
    """Find separate posterior cores in a narrow midline slab.

    Compare each core to the local plane perpendicular to the curved body
    centerline. This accounts for kyphosis/lordosis, unlike raw axial height.
    Require a unique match and stability at two physical erosion radii.
    """
    spacing = nib.affines.voxel_sizes(affine)
    x = affine[0,3] + np.arange(raw.shape[0])*affine[0,0]
    y = affine[1,3] + np.arange(raw.shape[1])*affine[1,1]
    z = affine[2,3] + np.arange(raw.shape[2])*affine[2,2]
    cx = np.interp(z,centers[:,2],centers[:,0])
    cy = np.interp(z,centers[:,2],centers[:,1])
    slab = ((np.abs(x[:,None,None]-cx[None,None,:]) < 6) &
            (y[None,:,None] < cy[None,None,:]-22) & (raw > 0) & (image >= -250))
    distance = ndi.distance_transform_edt(slab,sampling=spacing)
    derivatives = np.column_stack([np.gradient(centers[:,d],centers[:,2]) for d in (0,1)])
    tangents = np.column_stack([derivatives,np.ones(len(centers))])
    tangents /= np.linalg.norm(tangents,axis=1)[:,None]
    gaps = np.diff(centers[:,2])
    tolerances = .45*np.minimum(np.r_[gaps[0],gaps],np.r_[gaps,gaps[-1]])
    found = []
    vv = abs(np.linalg.det(affine[:3,:3]))
    for radius in (2.0,2.5):
        cc, _ = ndi.label(distance>radius,STRUCTURE)
        sizes = np.bincount(cc.ravel()); sizes[0]=0
        candidates = {}
        for cid in np.flatnonzero(sizes*vv >= 100):
            pos = np.argwhere(cc==cid)
            world = nib.affines.apply_affine(affine,pos.mean(0))
            differences = np.abs(np.sum((world-centers)*tangents,axis=1))
            ordered = np.argsort(differences); best = int(ordered[0])
            if differences[best] > tolerances[best] or differences[ordered[1]]-differences[best] < 4:
                continue
            if np.mean(image[tuple(pos.T)]>=130) < .5:
                continue
            candidates.setdefault(best,[]).append((int(cid),world,float(differences[best])))
        found.append((cc,candidates))
    markers = np.zeros(raw.shape,np.int16)
    records = []
    for index in range(9):
        left=found[0][1].get(index,[]); right=found[1][1].get(index,[])
        if len(left)!=1 or len(right)!=1: continue
        cid, world, error=left[0]
        stability=float(np.linalg.norm(world-right[0][1]))
        if stability>3: continue
        markers[found[0][0]==cid]=index+4
        records.append(dict(label=index+4,center_world_mm=world.tolist(),
                            plane_distance_mm=error,stability_mm=stability))
    return markers, records


def refine_thoracolumbar(ct, original, baseline, affine):
    """Return a copied label volume and an auditable decision report.

    The mask union is used only to locate separate thick vertebral-body cores.
    Their ordering, CT support and stability at two erosion radii must agree.
    Already-consistent cases are returned without any edits. Every existing
    non-target foreground voxel is locked throughout the operation.
    """
    report = dict(method='CT-supported body-core counting and geodesic partition',
                  target_ids=list(TARGETS), status='skipped', reasons=[])
    output = baseline.copy()
    if not np.array_equal(nib.aff2axcodes(affine), ('L','A','S')) and not np.array_equal(nib.aff2axcodes(affine), ('R','A','S')):
        # A general world-coordinate implementation is possible; this release
        # explicitly rejects non-axis-aligned layouts rather than guessing.
        report['reasons'].append('requires RAS/LAS voxel axes; reorient consistently first')
        return output, report
    linear = affine[:3, :3]
    if not np.allclose(linear, np.diag(np.diag(linear)), atol=1e-6):
        report['reasons'].append('oblique affine requires explicit resampling')
        return output, report
    spacing = nib.affines.voxel_sizes(affine)
    try:
        anchors = np.array([_largest(baseline == label) for label in (4,12)])
    except ValueError as exc:
        report['reasons'].append(str(exc)); return output, report
    anchor_world = nib.affines.apply_affine(affine, anchors)
    if anchor_world[1,2] <= anchor_world[0,2]:
        report['reasons'].append('L2/T6 anchor order invalid'); return output, report
    lo = np.maximum(0, np.floor(anchors.min(0)-np.array([65,85,45])/spacing).astype(int))
    hi = np.minimum(baseline.shape, np.ceil(anchors.max(0)+np.array([65,70,35])/spacing).astype(int))
    sl = tuple(slice(int(l),int(h)) for l,h in zip(lo,hi))
    local_affine = affine.copy(); local_affine[:3,3] = nib.affines.apply_affine(affine,lo)
    raw, old, image = original[sl], baseline[sl], ct[sl]
    # Use original foreground to recover components deleted by the old script.
    union = np.where(old > 0, old, raw).astype(np.uint8)
    foreground = (union > 0) & (image >= -250)
    # Detect cores on the original AI foreground, before morphology in the old
    # script can create bridges. Score identities separately using the baseline.
    distance = ndi.distance_transform_edt((raw > 0) & (image >= -250), sampling=spacing)
    trials = []
    accepted = None
    for radius in (3.,4.,5.,6.,7.):
        cc, components = _cores(distance, union, image, local_affine, radius, anchor_world)
        chain = _chain(components)
        trials.append({'radius_mm':radius, 'body_candidates':len(components), 'chain_found':chain is not None})
        if chain is None: continue
        _, next_components = _cores(distance, union, image, local_affine, radius+1, anchor_world)
        next_chain = _chain(next_components)
        if next_chain is None: continue
        stability = np.linalg.norm(np.array([c['center'] for c in chain[2]])-
                                   np.array([c['center'] for c in next_chain[2]]),axis=1)
        if np.max(stability) > 3: continue
        accepted = (radius, cc, components, chain, stability)
        break
    report['core_search'] = trials
    if accepted is None:
        report['reasons'].append('no stable nine-body L2..T6 chain; automatic relabelling withheld')
        return output, report
    radius, cc, components, (first,last,chain), stability = accepted
    report['core_radius_mm'] = radius
    report['max_core_center_shift_mm'] = float(stability.max())
    records = []
    for label, comp in zip(range(4,13), chain):
        counts = comp['counts']
        records.append(dict(expected_id=label, center_world_mm=comp['center'].tolist(),
                            dominant_input_id=int(counts.argmax()),
                            input_correct_fraction=float(counts[label]/counts.sum()),
                            input_label_counts=counts.tolist()))
    report['body_cores'] = records
    mismatch = sum(c['voxels'] - c['counts'][lab] for lab,c in zip(range(5,12),chain[1:-1]))
    total = sum(c['voxels'] for c in chain[1:-1])
    report['body_core_label_disagreement_fraction'] = float(mismatch/total)
    if mismatch/total < .05 and all(r['input_correct_fraction']>=.90 for r in records[1:-1]):
        report['status'] = 'unchanged_consistent'
        report['reasons'].append('body-core identities agree with existing labels')
        return output, report

    markers = np.zeros(raw.shape,dtype=np.int16)
    for label, comp in zip(range(4,13), chain): markers[cc == comp['cid']] = label
    # Include neighboring thick cores, so target labels cannot spread down/up
    # through bridges into the immediately neighboring vertebral bodies.
    if first > 0: markers[cc == components[first-1]['cid']] = 3
    if last+1 < len(components): markers[cc == components[last+1]['cid']] = 13
    posterior, posterior_records = _posterior_markers(
        raw,image,local_affine,np.array([c['center'] for c in chain]))
    markers[posterior>0]=posterior[posterior>0]
    report['posterior_cores'] = posterior_records
    # Geodesic competition follows physical bone connectivity. A penalty for
    # thin bridges discourages shortcuts through adjacent facet joints, while
    # preserving the connected pedicles and posterior elements of each body.
    distance = ndi.distance_transform_edt(foreground, sampling=spacing)
    cost = np.where(foreground, 1.0 + 8.0/(distance + .5)**2, np.inf)
    partition = np.zeros(raw.shape,dtype=np.uint8)
    best = np.full(raw.shape, np.inf)
    for label in np.unique(markers)[1:]:
        starts = np.argwhere(markers == label)
        solver = MCP_Geometric(cost, fully_connected=True, sampling=tuple(spacing))
        cumulative, _ = solver.find_costs(starts)
        closer = cumulative < best
        best[closer] = cumulative[closer]
        partition[closer] = label
    mutable = ((old>=5)&(old<=11)) | ((old==0)&(raw>=5)&(raw<=11))
    eligible = mutable & (partition>=4) & (partition<=12)
    new = output[sl]
    new[eligible] = partition[eligible]
    report['status'] = 'refined'
    report['relabelled_voxels'] = int(np.count_nonzero((new!=old)&(old>0)))
    report['restored_original_voxels'] = int(np.count_nonzero((new>0)&(old==0)))
    report['target_voxels_reassigned_to_anchors'] = int(np.count_nonzero(mutable & ((partition==4)|(partition==12))))
    report['unreachable_target_voxels'] = int(np.count_nonzero(mutable & (partition==0)))
    report['existing_non_target_voxels_changed'] = int(np.count_nonzero(
        (new!=old)&(old>0)&~((old>=5)&(old<=11))))
    report['reasons'].append('stable body cores disagree with labels; foreground partitioned between ordered cores')
    return output, report
