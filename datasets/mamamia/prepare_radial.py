"""Image-derived radial DCE inputs on an axial 384-mm, 320-pixel grid."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor,as_completed
from itertools import product
import json
import math
import os
from pathlib import Path
import tempfile

import h5py
import numpy as np
import SimpleITK as sitk

VERSION='mama-radial-LPS-fov384-grid320-z1-v1'
TRAIN_REDUCTIONS=(1,2,4,6,8)


def save_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temp.replace(path)


def signature(path):
    path=Path(path).resolve();s=path.stat()
    return dict(path=str(path),size_bytes=s.st_size,mtime_ns=s.st_mtime_ns)


def source_signatures(case):
    return {key:signature(case[key]) for key in ('precontrast_path','postcontrast_path','mask_path')}


def image_info(path):
    reader=sitk.ImageFileReader();reader.SetFileName(str(path));reader.ReadImageInformation()
    if reader.GetDimension()!=3:raise ValueError(f'Expected 3D image: {path}')
    return dict(size=list(reader.GetSize()),spacing=list(reader.GetSpacing()),
        origin=list(reader.GetOrigin()),direction=list(reader.GetDirection()))


def reference_geometry(info):
    size=np.asarray(info['size']);spacing=np.asarray(info['spacing']);direction=np.asarray(info['direction']).reshape(3,3)
    origin=np.asarray(info['origin']);center=origin+direction@(spacing*(size-1)/2)
    corners=np.array(list(product(*[[-.5,float(n)-.5] for n in size])))
    physical=origin+(corners*spacing)@direction.T
    depth=max(24,int(math.ceil(physical[:,2].max()-physical[:,2].min())))
    target_size=np.array([320,320,depth]);target_spacing=np.array([1.2,1.2,1.])
    target_origin=center-target_spacing*(target_size-1)/2
    return dict(size_xyz=target_size.tolist(),spacing_xyz=target_spacing.tolist(),origin_xyz=target_origin.tolist(),
        direction_xyz=np.eye(3).reshape(-1).tolist(),shape_dhw=[depth,320,320],orientation='LPS',field_of_view_xy_mm=[384.,384.])


def grid_image(geometry):
    image=sitk.Image(geometry['size_xyz'],sitk.sitkUInt8)
    image.SetSpacing(geometry['spacing_xyz']);image.SetOrigin(geometry['origin_xyz']);image.SetDirection(geometry['direction_xyz'])
    return image


def audit_case(case,cache):
    directory=Path(cache)/'labels'/case['exam_id'];path=directory/'geometry.json';sources=source_signatures(case)
    if path.exists():
        saved=json.loads(path.read_text())
        if saved['sources']!=sources or saved['version']!=VERSION:raise ValueError('Source changed after geometry preparation')
        if (directory/'label_mask.npy').exists():return dict(exam_id=case['exam_id'],complete=True)
    info=image_info(case['precontrast_path']);image_info(case['postcontrast_path']);geometry=reference_geometry(info)
    original=sitk.ReadImage(case['mask_path']);array=sitk.GetArrayFromImage(original)>0
    if not array.any():raise ValueError('Source annotation is empty')
    z,y,x=np.nonzero(array);coordinates=np.stack((x,y,z),axis=1)
    physical=np.asarray(original.GetOrigin())+(coordinates*np.asarray(original.GetSpacing()))@np.asarray(original.GetDirection()).reshape(3,3).T
    low=np.asarray(geometry['origin_xyz'])-np.asarray(geometry['spacing_xyz'])/2
    high=low+np.asarray(geometry['size_xyz'])*np.asarray(geometry['spacing_xyz'])
    outside=np.any((physical<low-1e-5)|(physical>high+1e-5),axis=1)
    if outside.any():raise ValueError(f'384-mm reference grid clips {int(outside.sum())} annotated source voxels; source extent {physical.min(0).tolist()} to {physical.max(0).tolist()}, reference {low.tolist()} to {high.tolist()}')
    binary=sitk.GetImageFromArray(array.astype(np.uint8));binary.CopyInformation(original)
    label=sitk.GetArrayFromImage(sitk.Resample(binary,grid_image(geometry),sitk.Transform(),sitk.sitkNearestNeighbor,0,sitk.sitkUInt8))
    if not label.any():raise ValueError('Annotation disappears on the requested grid')
    directory.mkdir(parents=True,exist_ok=True)
    temporary=directory/'label_mask.npy.tmp'
    with temporary.open('wb') as stream:np.save(stream,label)
    temporary.replace(directory/'label_mask.npy')
    save_json(path,dict(version=VERSION,sources=sources,reference=geometry,original_precontrast=info,
        original_mask_voxels=int(array.sum()),mask_voxels=int(label.sum()),original_mask_volume_mm3=float(array.sum()*np.prod(original.GetSpacing())),
        mask_volume_mm3=float(label.sum()*np.prod(geometry['spacing_xyz']))))
    return dict(exam_id=case['exam_id'],complete=True)


def audit(cases,cache,output,workers=8):
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(1)
    failures=[];completed=[]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures={pool.submit(audit_case,case,cache):case['exam_id'] for case in cases}
        for future in as_completed(futures):
            exam=futures[future]
            try:completed.append(future.result())
            except Exception as error:failures.append(dict(exam_id=exam,error=str(error)))
    save_json(output,dict(complete=not failures,examinations=len(cases),completed=len(completed),failures=failures,version=VERSION))
    if failures:raise RuntimeError(f'{len(failures)} MAMA-MIA geometry/annotation failures; see {output}')
    print(f'Validated geometry and labels for {len(completed)} subjects.',flush=True)


def source_volumes(case,cache):
    cache=Path(cache);directory=cache/'sources'/case['exam_id'];path=directory/'images.h5'
    metadata=json.loads((cache/'labels'/case['exam_id']/'geometry.json').read_text())
    if metadata['sources']!=source_signatures(case):raise ValueError('Original images or annotation changed')
    if path.exists():
        with h5py.File(path,'r') as f:
            if not f.attrs.get('complete') or json.loads(f.attrs['metadata_json'])!=metadata:raise ValueError('Stale source-volume cache')
        return path,metadata
    reference=grid_image(metadata['reference']);directory.mkdir(parents=True,exist_ok=True)
    fd,temp=tempfile.mkstemp(prefix='.source-',suffix='.h5',dir=directory);os.close(fd)
    try:
        with h5py.File(temp,'w') as out:
            for name,key in [('precontrast','precontrast_path'),('postcontrast','postcontrast_path')]:
                image=sitk.ReadImage(case[key],sitk.sitkFloat32)
                image=sitk.Resample(image,reference,sitk.Transform(),sitk.sitkBSpline,0.,sitk.sitkFloat32)
                values=sitk.GetArrayFromImage(image)
                if not np.isfinite(values).all() or not np.any(values):raise ValueError(f'Invalid resampled {name}')
                out.create_dataset(name,data=values,chunks=(1,320,320),compression='lzf',shuffle=True)
            out.attrs.update(complete=True,metadata_json=json.dumps(metadata,sort_keys=True))
        os.replace(temp,path)
    finally:Path(temp).unlink(missing_ok=True)
    return path,metadata


def full_measurements(source_path,cache,chunk=16):
    import torch
    from src.native_spokes.acquisition import operators
    with np.load(Path(cache)/'f100/geometry/R1.npz') as f:geometry={k:f[k] for k in f.files}
    ops=[op.cuda() for op in operators(geometry)]
    try:
        with h5py.File(source_path,'r') as source,torch.inference_mode():
            depth=len(source['precontrast']);measurements=np.empty((2,depth,96,640),dtype=np.complex64)
            for t,name in enumerate(('precontrast','postcontrast')):
                for start in range(0,depth,chunk):
                    stop=min(depth,start+chunk)
                    image=torch.as_tensor(source[name][start:stop],device='cuda',dtype=torch.complex64)[:,None]
                    values=ops[t].fwd(image).reshape(stop-start,96,640)
                    if not torch.isfinite(values).all():raise FloatingPointError('Nonfinite simulated measurements')
                    measurements[t,start:stop]=values.cpu().numpy()
        return measurements
    finally:
        for op in ops:op.clear_plan()


def condition_input(measurements,cache,reduction,percent=100,chunk=16):
    import torch
    from src.native_spokes.acquisition import operators,readout_slice,normalization_indices
    from src.native_spokes.sampling import selected_indices,WINDOW_STARTS
    from src.native_spokes.physics import fft2c
    with np.load(Path(cache)/f'f{percent}/geometry/R{reduction}.npz') as f:geometry={k:f[k] for k in f.files}
    selected=selected_indices(reduction);cut=readout_slice(percent)
    retained=np.stack([measurements[t][:,selected[t]-WINDOW_STARTS[t],cut] for t in range(2)])
    center=np.ascontiguousarray(retained[0,:,:,normalization_indices(percent)-cut.start])
    scale=float(np.sqrt(np.mean(np.abs(center)**2,dtype=np.float64)))
    if not np.isfinite(scale) or scale<=0:raise FloatingPointError('Invalid retained-precontrast normalization')
    ops=[op.cuda() for op in operators(geometry)];depth=measurements.shape[1]
    output=np.empty((depth,320,320),dtype=np.complex64)
    try:
        with torch.inference_mode():
            for start in range(0,depth,chunk):
                stop=min(depth,start+chunk)
                images=[ops[t].adj(torch.as_tensor(retained[t,start:stop]/scale,device='cuda').reshape(stop-start,1,-1)) for t in range(2)]
                values=fft2c(images[1]-images[0])[:,0]
                if not torch.isfinite(values).all():raise FloatingPointError('Nonfinite simulated combined input')
                output[start:stop]=values.cpu().numpy()
    finally:
        for op in ops:op.clear_plan()
    return output,dict(normalization_scale=scale,normalization_reference_indices=normalization_indices(percent).tolist(),
        selected_original_spokes=selected.tolist(),retained_spokes_per_window=len(selected[0]),
        retained_readout_samples=cut.stop-cut.start,effective_spoke_reduction=96/len(selected[0]))


def prepare(case,cache,output):
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(min(8,int(os.environ.get('NATIVE_NUM_THREADS','8'))))
    cache=Path(cache);source,source_metadata=source_volumes(case,cache)
    states=[];missing=[]
    for r in TRAIN_REDUCTIONS:
        directory=cache/'f100'/case['exam_id'];directory.mkdir(parents=True,exist_ok=True)
        label=directory/'label_mask.npy'
        if not label.exists():label.symlink_to((cache/'labels'/case['exam_id']/'label_mask.npy').resolve())
        path=directory/f'R{r}.h5'
        if path.exists():
            with h5py.File(path,'r') as f:
                meta=json.loads(f.attrs['metadata_json'])
                if not f.attrs.get('complete') or meta['sources']!=source_metadata['sources'] or meta['version']!=VERSION:raise ValueError('Stale radial input')
        else:missing.append((r,path))
        states.append(str(path))
    if missing:
        measured=full_measurements(source,cache)
        for r,path in missing:
            values,metadata=condition_input(measured,cache,r)
            metadata.update(version=VERSION,dataset='mama_mia',acquisition_origin='image-derived radial simulation',
                exam_id=case['exam_id'],person_id=case['person_id'],sources=source_metadata['sources'],
                reference_grid=source_metadata['reference'],spoke_reduction=r,readout_percent=100)
            temp=path.with_suffix('.h5.tmp')
            try:
                with h5py.File(temp,'w') as f:
                    f.create_dataset('combined_kspace',data=values,chunks=(1,320,320),compression='lzf',shuffle=True)
                    f.attrs.update(complete=True,preprocessing_version=VERSION,metadata_json=json.dumps(metadata,sort_keys=True))
                temp.replace(path)
            finally:temp.unlink(missing_ok=True)
    save_json(output,dict(complete=True,exam_id=case['exam_id'],version=VERSION,source=str(source),states=states,
        shape_dhw=source_metadata['reference']['shape_dhw']))
    print(f'{case["exam_id"]}: five training conditions complete',flush=True)


def validate(cases,cache,prepared,output):
    from src.native_spokes.acquisition import normalization_indices
    from src.native_spokes.sampling import selected_indices
    cache=Path(cache);failures=[]
    for case in cases:
        try:
            receipt=json.loads((Path(prepared)/f'{case["exam_id"]}.json').read_text())
            if not receipt['complete']:raise ValueError('Incomplete case')
            label=np.load(cache/'labels'/case['exam_id']/'label_mask.npy',mmap_mode='r')
            if not label.any() or list(label.shape)!=receipt['shape_dhw']:raise ValueError('Target/grid mismatch')
            for r in TRAIN_REDUCTIONS:
                with h5py.File(cache/'f100'/case['exam_id']/f'R{r}.h5','r') as f:
                    meta=json.loads(f.attrs['metadata_json']);values=f['combined_kspace']
                    if not f.attrs.get('complete') or values.shape!=label.shape or values.dtype!=np.complex64:raise ValueError('Invalid input shape/type')
                    if meta['sources']!=source_signatures(case) or meta['selected_original_spokes']!=selected_indices(r).tolist() or meta['normalization_reference_indices']!=normalization_indices(100).tolist():raise ValueError('Input provenance/selection differs')
                    for z in (0,len(label)//2,len(label)-1):
                        if not np.isfinite(values[z]).all():raise ValueError('Nonfinite prepared data')
        except Exception as error:failures.append(dict(exam_id=case['exam_id'],error=str(error)))
    save_json(output,dict(complete=not failures,examinations=len(cases),conditions_per_exam=5,
        cache_states=len(cases)*5,version=VERSION,failures=failures))
    if failures:raise RuntimeError(f'{len(failures)} incomplete/invalid subjects; see {output}')
