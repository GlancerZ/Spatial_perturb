"""Inspect experiment-matched image axes/calibration and decoded-point dimensions."""
import io
import json
from pathlib import Path
import zipfile
import numpy as np
from scipy.io import loadmat
import tifffile
from roifile import ImagejRoi

root = Path('data/subcellular_data_agent/raw_public_20260922/seqfish_plus_nih3t3_zenodo_2669683')
report = {}
for run in (1, 2):
    with zipfile.ZipFile(root / f'DAPI_experiment{run}.zip') as archive:
        name = next(n for n in archive.namelist() if n.lower().endswith('.tif'))
        with archive.open(name) as raw, tifffile.TiffFile(io.BytesIO(raw.read())) as image:
            item = {'member': name, 'series': [{'shape': s.shape, 'axes': s.axes} for s in image.series],
                    'ome_metadata': image.ome_metadata, 'imagej_metadata': image.imagej_metadata,
                    'first_description': image.pages[0].description}
    with zipfile.ZipFile(root / f'ROIs_Experiment{run}_NIH3T3.zip') as archive:
        names = [n for n in archive.namelist() if n.endswith('.roi')][:3]
        item['roi_examples'] = []
        for name in names:
            roi = ImagejRoi.frombytes(archive.read(name))
            coords = roi.coordinates()
            item['roi_examples'].append({'name': name, 'position': roi.position,
                                         'c_position': roi.c_position, 'z_position': roi.z_position,
                                         't_position': roi.t_position,
                                         'bbox_xy': [coords.min(axis=0).tolist(), coords.max(axis=0).tolist()]})
    with zipfile.ZipFile(root / 'seqFISH_NIH3T3_point_locations.zip') as archive:
        arrays = loadmat(io.BytesIO(archive.read(f'RNA_locations_run_{run}.mat')))['tot']
        item['mat_shape'] = arrays.shape
        examples = []
        for cell in range(min(arrays.shape[1], 4)):
            nonempty = [a for a in arrays[0, cell, :] if a.size]
            points = np.concatenate(nonempty, axis=0) if nonempty else np.empty((0, 0))
            examples.append({'cell_index': cell, 'nonempty_genes': len(nonempty),
                             'points_shape': points.shape,
                             'points_min': points.min(axis=0).tolist() if len(points) else [],
                             'points_max': points.max(axis=0).tolist() if len(points) else [],
                             'point_examples': points[:5].tolist()})
        item['point_examples'] = examples
    report[f'run{run}'] = item
output = Path('outputs/seqfish_image_metadata_20260922.json')
if output.exists():
    raise FileExistsError(output)
output.write_text(json.dumps(report, indent=2, default=str))
print(f'Written {output}', flush=True)
