"""Image-only PSF selection, local calibration and native-exposure deblending.

No catalog or injection truth enters these functions. Empirical scores are NOT
Gaussian-tail probabilities. Correlation diagnostics do not imply whitening.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy.ndimage import binary_dilation, distance_transform_edt, shift, zoom

from astr_ir.background.processor import neighbor_difference_noise
from astr_ir.flicker.processor import robust_std


@dataclass(frozen=True)
class DetectionConfig:
    psf_mode: str = 'auto'
    psf_size: int = 25
    noise_box: int = 64
    min_noise_pixels: int = 128
    threshold: float = 5.0
    min_distance: int = 2
    iterations: int = 3
    max_sources: int = 1000
    max_group: int = 12
    crowded_background_box: int = 64
    min_coverage: float = 0.8
    local_noise: bool = True

    def validate(self):
        if self.psf_mode not in {'auto','gaussian','elliptical','moffat','empirical'}:
            raise ValueError('Unknown PSF mode')
        if self.psf_size < 15 or self.psf_size % 2 != 1:
            raise ValueError('psf_size must be odd and >=15')
        if self.noise_box < 16 or self.min_noise_pixels < 32:
            raise ValueError('Insufficient noise mesh size/support')
        if self.threshold <= 0 or self.min_distance < 1 or not 1 <= self.iterations <= 5:
            raise ValueError('Invalid detection thresholds/iteration limit')
        if self.max_sources < 1 or not 1 <= self.max_group <= 24 or not 0 < self.min_coverage <= 1:
            raise ValueError('Invalid source/group/coverage limits')
        if self.crowded_background_box < 32:
            raise ValueError('Crowded background cells must span at least 32 pixels')


def shifted_kernel(kernel, dy=0., dx=0.):
    """Shift a PSF MODEL, not science pixels; preserve unit integrated flux."""
    kernel = np.asarray(kernel,float)
    if kernel.ndim != 2 or any(n % 2 != 1 for n in kernel.shape) or not np.isfinite(kernel).all() or np.any(kernel < 0) or kernel.sum() <= 0:
        raise ValueError('PSF must be finite, nonnegative, odd-sized, and nonzero')
    moved = shift(kernel,(dy,dx),order=1,mode='constant',cval=0,prefilter=False)
    return moved/moved.sum()


def analytic_psf(size, sx, sy=None, angle=0., beta=None):
    yy,xx = np.mgrid[:size,:size] - size//2
    sy = sx if sy is None else sy
    u = xx*np.cos(angle)+yy*np.sin(angle)
    v = -xx*np.sin(angle)+yy*np.cos(angle)
    radius = (u/sx)**2+(v/sy)**2
    # sx/sy are Gaussian-equivalent core scales; beta controls wings.
    kernel = np.exp(-radius/2) if beta is None else (1+radius/(2*beta))**(-beta)
    return kernel/kernel.sum()


def _fit_profile(target, mode, sigma):
    p = np.array([sigma,sigma,0.,3.5],float)
    def model(q):
        return analytic_psf(len(target),q[0],q[1],q[2],q[3] if mode=='moffat' else None)
    best = float(np.sum((model(p)-target)**2))
    for step in (.5,.25,.1,.04):
        for _ in range(2):
            for axis in range(4 if mode=='moffat' else 3):
                for direction in (-1,1):
                    trial=p.copy(); trial[axis]+=direction*step
                    trial=np.clip(trial,[.7,.7,-np.pi,1.5],[5.,5.,np.pi,8.])
                    error=float(np.sum((model(trial)-target)**2))
                    if error < best: p,best=trial,error
    return model(p)


def estimate_psf(image, valid, stars, config=None):
    """Select shape using disjoint automatic-star fit/validation subsets.

Stars have x,y,width,flux columns from inspect_frame, never catalog positions.
Empirical models require >=8 usable isolated stars. Analytic shape fitting
requires >=4; otherwise retain an explicitly flagged circular fallback.
"""
    config=config or DetectionConfig(); config.validate()
    size=config.psf_size; radius=size//2
    sigma=float(np.median(stars[:,2])) if len(stars)>=3 else 2.5
    circular=analytic_psf(size,sigma)
    samples=[]
    yy,xx=np.mgrid[-radius:radius+1,-radius:radius+1]
    annulus=np.hypot(xx,yy)>=radius*.8
    for i,(x,y,_,_) in enumerate(stars):
        distances=np.hypot(stars[:,0]-x,stars[:,1]-y); distances[i]=np.inf
        if distances.min(initial=np.inf) < radius*1.5: continue
        ix,iy=int(round(x)),int(round(y))
        if min(ix,iy)<radius or iy+radius>=image.shape[0] or ix+radius>=image.shape[1]: continue
        cut=np.asarray(image[iy-radius:iy+radius+1,ix-radius:ix+radius+1],float)
        good=valid[iy-radius:iy+radius+1,ix-radius:ix+radius+1] & np.isfinite(cut)
        if good.mean()<.95 or (annulus & good).sum()<24: continue
        level=np.median(cut[good & annulus])
        support=shift(good.astype(float),(iy-y,ix-x),order=1,mode='constant',cval=0,prefilter=False)
        numerator=shift(np.where(good,cut-level,0),(iy-y,ix-x),order=1,mode='constant',cval=0,prefilter=False)
        centered=np.divide(numerator,support,out=np.zeros_like(cut),where=support>=.5)
        total=centered.sum()
        if total<=0 or centered.max()<10*robust_std(cut[good & annulus]): continue
        samples.append(centered/total)
    info=dict(psf_mode='gaussian',psf_samples=len(samples),psf_fallback=False,psf_sigma=sigma)
    if config.psf_mode=='gaussian': return circular,info
    if len(samples)<4:
        info.update(psf_fallback=True,psf_reason='fewer than four isolated usable stars')
        return circular,info
    train=np.median(samples[::2],axis=0)
    train=np.maximum(train,0); train/=train.sum()
    candidates={'gaussian':circular,'elliptical':_fit_profile(train,'elliptical',sigma),
                'moffat':_fit_profile(train,'moffat',sigma)}
    if len(samples)>=8: candidates['empirical']=train
    errors={name:float(np.median([np.sum((sample-kernel)**2) for sample in samples[1::2]]))
            for name,kernel in candidates.items()}
    best=min(errors,key=errors.get) if config.psf_mode=='auto' else config.psf_mode
    if best not in candidates:
        best=min(errors,key=errors.get); info.update(psf_fallback=True,psf_reason='empirical requires eight usable stars')
    if config.psf_mode=='auto' and errors[best]>.95*errors['gaussian']: best='gaussian'
    info.update(psf_mode=best,psf_validation_error=errors[best],psf_circular_error=errors['gaussian'])
    return candidates[best],info


def _expand_grid(grid, shape):
    good=np.isfinite(grid)
    if not good.any(): raise ValueError('No valid local noise cells')
    nearest=distance_transform_edt(~good,return_distances=False,return_indices=True)
    filled=grid[tuple(nearest)]
    return zoom(filled,(shape[0]/grid.shape[0],shape[1]/grid.shape[1]),order=1,mode='nearest',prefilter=False)[:shape[0],:shape[1]]


def automatic_noise_mask(image, valid, radius=8):
    values=image[valid & np.isfinite(image)]
    center=np.median(values); scale=robust_std(values)
    seeds=valid & np.isfinite(image) & (np.abs(image-center)>7*max(scale,1e-12))
    # Exclude either sign: bright wings and filter sidelobes both bias noise.
    return valid & ~binary_dilation(seeds,iterations=radius)


def local_noise_map(image, valid, fallback, config):
    blank=automatic_noise_mask(image,valid)
    h,w=image.shape; box=config.noise_box
    grid=np.full(((h+box-1)//box,(w+box-1)//box),np.nan)
    for gy,y in enumerate(range(0,h,box)):
        for gx,x in enumerate(range(0,w,box)):
            good=blank[y:y+box,x:x+box]; cut=image[y:y+box,x:x+box]
            if good.sum()>=config.min_noise_pixels:
                value=neighbor_difference_noise(cut,~good)
                if np.isfinite(value) and value>0: grid[gy,gx]=value
    if not np.isfinite(grid).any(): return np.full(image.shape,fallback),dict(noise_mesh_fallback=True)
    noise=np.maximum(_expand_grid(grid,image.shape),.5*fallback)
    return noise,dict(noise_mesh_fallback=False,noise_mesh_valid=int(np.isfinite(grid).sum()))


def local_calibration(nominal, valid, config):
    blank=automatic_noise_mask(nominal,valid)
    center=float(np.median(nominal[blank])) if blank.any() else float(np.median(nominal[valid]))
    scale=robust_std(nominal[blank]) if blank.any() else robust_std(nominal[valid])
    if not np.isfinite(scale) or scale<=0: raise ValueError('Degenerate empirical score')
    h,w=nominal.shape; box=config.noise_box
    centers=np.full(((h+box-1)//box,(w+box-1)//box),np.nan); scales=centers.copy()
    if config.local_noise:
        for gy,y in enumerate(range(0,h,box)):
            for gx,x in enumerate(range(0,w,box)):
                values=nominal[y:y+box,x:x+box][blank[y:y+box,x:x+box]]
                if len(values)>=config.min_noise_pixels:
                    s=robust_std(values)
                    if s>0: centers[gy,gx]=np.median(values); scales[gy,gx]=max(1.,s)
    fallback=not np.isfinite(scales).any()
    center_map=np.full((h,w),center) if fallback else _expand_grid(centers,(h,w))
    scale_map=np.full((h,w),max(1.,scale)) if fallback else np.maximum(1.,_expand_grid(scales,(h,w)))
    score=np.where(valid,(nominal-center_map)/scale_map,np.nan)
    return score,center_map,scale_map,dict(local_calibration_fallback=fallback,
        empirical_noise_scale=float(np.median(scale_map[valid])),nominal_center=center,
        blank_fraction=float(blank.mean()),nominal_scale_floor=1.0)


def blank_correlations(image, valid):
    blank=automatic_noise_mask(image,valid)
    centered=np.where(blank,image-np.median(image[blank]) if blank.any() else 0,0)
    result={}
    for axis,name in ((0,'y'),(1,'x')):
        a,b=(centered[:-1],centered[1:]) if axis==0 else (centered[:,:-1],centered[:,1:])
        good=(blank[:-1]&blank[1:]) if axis==0 else (blank[:,:-1]&blank[:,1:])
        denom=np.sqrt(np.sum(a[good]**2)*np.sum(b[good]**2))
        result[f'blank_lag1_{name}']=float(np.sum(a[good]*b[good])/denom) if denom>0 else None
    result['whitening_applied']=False
    return result


@dataclass
class Exposure:
    image: np.ndarray
    valid: np.ndarray
    noise: np.ndarray | float
    offset: np.ndarray
    throughput: float
    psf: np.ndarray


def source_patch(shape, psf, x, y):
    ix,iy=int(round(x)),int(round(y)); ry,rx=np.array(psf.shape)//2
    kernel=shifted_kernel(psf,y-iy,x-ix)
    xlo,xhi=max(0,ix-rx),min(shape[1],ix+rx+1)
    ylo,yhi=max(0,iy-ry),min(shape[0],iy+ry+1)
    if xlo>=xhi or ylo>=yhi: return (slice(0,0),slice(0,0)),kernel[:0,:0]
    return (slice(ylo,yhi),slice(xlo,xhi)),kernel[ylo-iy+ry:yhi-iy+ry,xlo-ix+rx:xhi-ix+rx]


def render_sources(exposure, positions, fluxes):
    model=np.zeros_like(exposure.image,dtype=float)
    for (x,y),flux in zip(positions,fluxes):
        slices,kernel=source_patch(model.shape,exposure.psf,x-exposure.offset[1],y-exposure.offset[0])
        model[slices]+=flux*exposure.throughput*kernel
    return model


def _small_inverse(matrix):
    """Pivoted scalar Gauss-Jordan for bounded candidate systems; no threaded BLAS."""
    n=len(matrix); work=np.column_stack((matrix.copy(),np.eye(n)))
    scale=max(float(np.max(np.abs(matrix))),1e-30)
    for k in range(n):
        pivot=k+int(np.argmax(np.abs(work[k:,k])))
        if abs(work[pivot,k])<1e-9*scale: raise ValueError('Degenerate source group')
        work[[k,pivot]]=work[[pivot,k]]
        work[k]/=work[k,k]
        for row in range(n):
            if row!=k: work[row]-=work[row,k]*work[k]
    return work[:,n:]


def _fit_group(exposures, positions):
    count=len(positions); normal=np.zeros((count,count)); rhs=np.zeros(count)
    for exposure in exposures:
        native=positions-np.asarray(exposure.offset)[::-1]
        radius=max(exposure.psf.shape)//2
        lo=np.maximum(np.floor(native.min(axis=0)).astype(int)-radius,0)
        hi=np.minimum(np.ceil(native.max(axis=0)).astype(int)+radius+1,exposure.image.shape[::-1])
        if np.any(hi<=lo): continue
        cut=exposure.image[lo[1]:hi[1],lo[0]:hi[0]]
        valid=exposure.valid[lo[1]:hi[1],lo[0]:hi[0]] & np.isfinite(cut)
        noise=np.broadcast_to(exposure.noise,exposure.image.shape)[lo[1]:hi[1],lo[0]:hi[0]]
        if valid.sum()<count+10: continue
        weight=np.where(valid,1/noise**2,0); total=weight.sum()
        templates=[]
        for x,y in native-lo:
            template=np.zeros_like(cut,float)
            slices,kernel=source_patch(cut.shape,exposure.psf,x,y)
            template[slices]=kernel*exposure.throughput
            template-=np.sum(template*weight)/total  # marginalize per-frame constant background
            templates.append(template)
        data=np.where(valid,cut,0)
        for i in range(count):
            rhs[i]+=np.sum(templates[i]*data*weight)
            for j in range(i+1):
                value=np.sum(templates[i]*templates[j]*weight)
                normal[i,j]+=value
                if i!=j: normal[j,i]+=value
    covariance=_small_inverse(normal)
    flux=np.sum(covariance*rhs[None,:],axis=1)
    errors=np.sqrt(np.maximum(0,np.diag(covariance)))
    return flux,errors,float(np.sum(flux*rhs))


def fit_native_sources(exposures, positions, config, *, return_diagnostics=False):
    """Group overlaps and refit ORIGINAL native valid pixels, not residual pixels."""
    positions=np.asarray(positions,float).reshape(-1,2).copy()
    fluxes=np.zeros(len(positions)); errors=np.full(len(positions),np.inf)
    flags=np.full(len(positions),'ok',dtype=object)
    details=[None]*len(positions)
    unvisited=set(range(len(positions)))
    linking=max(exposures[0].psf.shape)  # group all overlapping PSF footprints
    while unvisited:
        group={min(unvisited)}; frontier=list(group); unvisited-=group
        while frontier:
            i=frontier.pop()
            neighbors={j for j in unvisited if np.linalg.norm(positions[i]-positions[j])<linking}
            group|=neighbors; unvisited-=neighbors; frontier.extend(neighbors)
        indices=sorted(group)
        locations=positions[indices].copy()
        large=len(indices)>config.max_group
        detail=dict(group_size=len(indices),fit_method='sparse_joint_local_background' if large else 'dense_joint',
                    position_sweeps_accepted=0)
        for index in indices: details[index]=detail
        try:
            if large:
                from .crowded_fit import fit_crowded_group
                locations,flux,err,refinement=fit_crowded_group(exposures,locations,config.crowded_background_box)
                detail.update(refinement)
                positions[indices]=locations; fluxes[indices]=flux; errors[indices]=err
                flags[np.asarray(indices)[flux<=0]]='nonpositive_flux'
                continue
            flux,err,best=_fit_group(exposures,locations)
            # Bounded subpixel refinement around blind peaks; no truth/catalog input.
            for step in (.5,.25):
                for i in range(len(locations)):
                    for axis in range(2):
                        for direction in (-1,1):
                            trial=locations.copy(); trial[i,axis]+=direction*step
                            f,e,improvement=_fit_group(exposures,trial)
                            if improvement>best and np.all(f>=0):
                                locations,flux,err,best=trial,f,e,improvement
            fluxes[indices]=flux; errors[indices]=err; positions[indices]=locations
            for i,f in zip(indices,flux):
                if f<=0: flags[i]='nonpositive_flux'
        except ValueError:
            flags[indices]='degenerate_group'
    result=(positions,fluxes,errors,flags)
    return (*result,details) if return_diagnostics else result


def analyze_exposures(exposures, config=None, *, fit_sources=True):
    """Native matched-filter candidates followed by bounded residual discovery."""
    from .blind_joint import exposure_statistic, detect_blind
    config=config or DetectionConfig(); config.validate()
    if not exposures: raise ValueError('No exposures')
    shape=exposures[0].image.shape
    positions=np.empty((0,2)); fluxes=np.empty(0); first=None; rounds=[]
    for iteration in range(config.iterations):
        numerator=np.zeros(shape); information=np.zeros(shape); coverage=np.zeros(shape,int)
        for exposure in exposures:
            residual=exposure.image-render_sources(exposure,positions,fluxes) if len(positions) else exposure.image
            n,d,good=exposure_statistic(residual,exposure.valid,exposure.noise,2.5,
                exposure.offset,exposure.throughput,psf_kernel=exposure.psf)
            numerator+=n; information+=d; coverage+=good
        good=(coverage>=np.ceil(config.min_coverage*len(exposures))) & (information>0)
        nominal=np.divide(numerator,np.sqrt(information),out=np.full(shape,np.nan),where=good)
        if first is None:
            score,center,scale,diagnostic=local_calibration(nominal,good,config)
            first=dict(score=score,nominal=nominal,center=center,scale=scale,information=information,
                       coverage=coverage,valid=good,diagnostic=diagnostic,
                       flux=np.divide(numerator,information,out=np.full(shape,np.nan),where=good))
        else:
            # Freeze the initial calibration. Never amplify residuals by shrinking their RMS.
            score=np.where(good,(nominal-first['center'])/first['scale'],np.nan)
        if not fit_sources or len(positions)>=config.max_sources: break
        peaks=detect_blind(score,good,config.threshold,config.min_distance)
        fresh=[]
        for row in peaks.itertuples():
            point=np.array([row.x,row.y],float)
            if len(positions) and np.min(np.linalg.norm(positions-point,axis=1))<config.min_distance: continue
            fresh.append(point)
            if len(positions)+len(fresh)>=config.max_sources: break
        rounds.append(dict(iteration=iteration+1,new_candidates=len(fresh)))
        if not fresh: break
        positions=np.concatenate((positions,np.asarray(fresh)))
        positions,fluxes,errors,flags,details=fit_native_sources(exposures,positions,config,return_diagnostics=True)
        fluxes=np.where(flags=='ok',fluxes,0.)
    rows=[]
    if len(positions):
        for i,((x,y),flux,error,flag) in enumerate(zip(positions,fluxes,errors,flags)):
            ix,iy=int(np.clip(round(x),0,shape[1]-1)),int(np.clip(round(y),0,shape[0]-1))
            snr=flux/error if error>0 else 0.
            calibrated=snr/first['scale'][iy,ix]
            rows.append(dict(x=x,y=y,flux=flux,flux_error_nominal=error,fit_snr_nominal=snr,
                snr_empirical=calibrated,fit_flag=flag,accepted=bool(flag=='ok' and calibrated>=config.threshold and first['valid'][iy,ix]),
                initial_score=float(first['score'][iy,ix]),**details[i]))
    first['sources']=pd.DataFrame(rows,columns=['x','y','flux','flux_error_nominal','fit_snr_nominal',
        'snr_empirical','fit_flag','accepted','initial_score','group_size','fit_method','position_sweeps_accepted'])
    first['diagnostic'].update(rounds=rounds,whitening_applied=False,
        significance='Local central-distribution calibration; NOT calibrated false-alarm probability',
        group_limit=config.max_group,candidate_limit=config.max_sources,
        group_limit_semantics='dense-fit crossover, not a candidate rejection limit',
        crowded_background_box=config.crowded_background_box,
        crowded_candidates=sum(d['group_size']>config.max_group for d in details) if len(positions) else 0,
        largest_fitted_group=max((d['group_size'] for d in details),default=0) if len(positions) else 0)
    return first
