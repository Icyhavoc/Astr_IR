"""Sparse native-pixel fits for crowded connected candidate groups.

All sources remain in ONE flux/covariance solve. Disjoint background cells
marginalize local constants without counting a science pixel more than once.
Local residual windows are used only to propose positions; final photometry
and covariance always come from the original valid exposures.
"""
from __future__ import annotations

import numpy as np
from scipy.sparse import coo_matrix


def group_bounds(exposures, positions):
    bounds=[]
    for exposure in exposures:
        native=positions-np.asarray(exposure.offset)[::-1]
        radius=np.asarray(exposure.psf.shape)[::-1]//2+2
        lo=np.maximum(np.floor(native.min(axis=0)).astype(int)-radius,0)
        hi=np.minimum(np.ceil(native.max(axis=0)).astype(int)+radius+1,exposure.image.shape[::-1])
        bounds.append((lo,hi))
    return bounds


def sparse_group_system(exposures, positions, bounds, background_box=64):
    """Schur complement of source templates and disjoint local backgrounds.

    Sparse products have only PSF-footprint support. Dense storage is O(N²)
    for the source covariance, not O(N * number_of_science_pixels).
    A None background_box uses one constant and supports dense-reference tests.
    """
    from .weak_detection import source_patch
    count=len(positions); normal=np.zeros((count,count)); rhs=np.zeros(count)
    for exposure,(lo,hi) in zip(exposures,bounds):
        if np.any(hi<=lo): continue
        ys,xs=slice(lo[1],hi[1]),slice(lo[0],hi[0])
        image=np.asarray(exposure.image[ys,xs],float)
        good=exposure.valid[ys,xs] & np.isfinite(image)
        noise=np.broadcast_to(exposure.noise,exposure.image.shape)[ys,xs]
        if np.any(good & (~np.isfinite(noise) | (noise<=0))):
            raise ValueError('Invalid native noise in crowded fit')
        if good.sum()<count+10: continue
        weight=np.divide(1.,noise**2,out=np.zeros_like(image),where=good).ravel()
        data=np.where(good,image-np.median(image[good]),0).ravel()
        height,width=image.shape
        row_indices=[]; columns=[]; values=[]
        native=positions-np.asarray(exposure.offset)[::-1]-lo
        for index,(x,y) in enumerate(native):
            slices,kernel=source_patch(image.shape,exposure.psf,x,y)
            yy,xx=np.mgrid[slices[0],slices[1]]
            rows=(yy*width+xx).ravel()
            row_indices.extend(rows); columns.extend([index]*len(rows))
            values.extend((kernel*exposure.throughput).ravel())
        templates=coo_matrix((values,(row_indices,columns)),shape=(image.size,count)).tocsr()
        normal+=(templates.T @ templates.multiply(weight[:,None])).toarray()
        rhs+=np.asarray(templates.T @ (weight*data)).ravel()
        if background_box is None:
            cells=np.zeros(image.size,dtype=int)
        else:
            yy,xx=np.mgrid[lo[1]:hi[1],lo[0]:hi[0]]
            gy=yy//background_box-lo[1]//background_box
            gx=xx//background_box-lo[0]//background_box
            cells=(gy*(int(gx.max())+1)+gx).ravel()
        cell_count=int(cells.max())+1
        total_weight=np.bincount(cells,weights=weight,minlength=cell_count)
        total_data=np.bincount(cells,weights=weight*data,minlength=cell_count)
        nuisance=coo_matrix((weight,(np.arange(image.size),cells)),shape=(image.size,cell_count)).tocsr()
        coupling=(templates.T @ nuisance).toarray()
        for cell in np.flatnonzero(total_weight>0):
            vector=coupling[:,cell]
            normal-=np.outer(vector,vector)/total_weight[cell]
            rhs-=vector*(total_data[cell]/total_weight[cell])
    return (normal+normal.T)*.5,rhs


def solve_group_system(normal, rhs):
    from .weak_detection import _small_inverse
    diagonal=np.diag(normal)
    if not np.isfinite(normal).all() or not np.isfinite(rhs).all() or np.any(diagonal<=0):
        raise ValueError('Insufficient information for crowded fit')
    scale=np.sqrt(diagonal)
    covariance=_small_inverse(normal/scale[:,None]/scale[None,:])/scale[:,None]/scale[None,:]
    flux=np.sum(covariance*rhs[None,:],axis=1)
    variance=np.diag(covariance)
    if np.any(variance<=0) or not np.isfinite(flux).all():
        raise ValueError('Degenerate crowded covariance')
    return flux,np.sqrt(variance),float(np.sum(flux*rhs))


def _local_exposures(exposures, residuals, point, flux):
    from .weak_detection import Exposure,render_sources
    local=[]
    for exposure,residual in zip(exposures,residuals):
        native=point-np.asarray(exposure.offset)[::-1]
        radius=np.asarray(exposure.psf.shape)[::-1]//2+2
        lo=np.maximum(np.floor(native).astype(int)-radius,0)
        hi=np.minimum(np.ceil(native).astype(int)+radius+1,exposure.image.shape[::-1])
        if np.any(hi<=lo): continue
        ys,xs=slice(lo[1],hi[1]),slice(lo[0],hi[0])
        item=Exposure(residual[ys,xs].copy(),exposure.valid[ys,xs],
            np.broadcast_to(exposure.noise,exposure.image.shape)[ys,xs],
            np.asarray(exposure.offset)+lo[::-1],exposure.throughput,exposure.psf)
        item.image+=render_sources(item,point[None],[flux])
        local.append(item)
    return local


def _update_residual(exposures, residuals, point, flux):
    from .weak_detection import source_patch
    for exposure,residual in zip(exposures,residuals):
        slices,kernel=source_patch(residual.shape,exposure.psf,point[0]-exposure.offset[1],point[1]-exposure.offset[0])
        residual[slices]+=flux*exposure.throughput*kernel


def fit_crowded_group(exposures, positions, background_box=64):
    from .weak_detection import _fit_group,render_sources
    locations=np.asarray(positions,float).copy()
    # Freeze the likelihood domain and nuisance cells for all position proposals.
    bounds=group_bounds(exposures,locations)
    system=sparse_group_system(exposures,locations,bounds,background_box)
    flux,errors,best=solve_group_system(*system)
    accepted_sweeps=0
    for step in (.5,.25):
        proposed=locations.copy(); conditional_flux=flux.copy()
        residuals=[np.asarray(e.image,float)-render_sources(e,locations,flux) for e in exposures]
        for index,point in enumerate(locations):
            if flux[index]<=0: continue
            local=_local_exposures(exposures,residuals,point,conditional_flux[index])
            if not local: continue
            try:
                f,_,value=_fit_group(local,point[None])
                candidate=point.copy(); candidate_flux=float(f[0])
                for axis in range(2):
                    for direction in (-1,1):
                        trial=candidate.copy(); trial[axis]+=direction*step
                        trial_flux,_,improvement=_fit_group(local,trial[None])
                        if improvement>value and trial_flux[0]>0:
                            candidate,candidate_flux,value=trial,float(trial_flux[0]),improvement
                _update_residual(exposures,residuals,point,conditional_flux[index])
                _update_residual(exposures,residuals,candidate,-candidate_flux)
                proposed[index],conditional_flux[index]=candidate,candidate_flux
            except ValueError:
                continue  # Keep this source in the full original-pixel solve.
        try:
            candidate_system=sparse_group_system(exposures,proposed,bounds,background_box)
            f,e,value=solve_group_system(*candidate_system)
        except ValueError:
            continue
        # Local conditional fits are proposals only; they may never worsen the
        # SAME full-group native-pixel objective or supply conditional errors.
        if value>best+1e-10*max(abs(best),1.):
            locations,flux,errors,best=proposed,f,e,value
            accepted_sweeps+=1
    return locations,flux,errors,dict(position_sweeps_accepted=accepted_sweeps)
