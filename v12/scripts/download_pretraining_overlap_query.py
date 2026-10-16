"""Download public dataset viewer responses for cohort-overlap provenance."""
import json
from pathlib import Path
import urllib.request
import urllib.parse
p=Path('resources/spatialformer')
protocol=json.loads((p/'protocol.json').read_text())
names=sorted(set(protocol['samples'])|set(protocol['old_to_current_sample_names']))
where=' OR '.join('"Sample_Names" = '+"'"+name+"'" for name in names)
params=dict(dataset='TerminatorJ/xenium_5k_pandavid_dataset_v2',config='default',split='train',where=where,length=1)
url='https://datasets-server.huggingface.co/filter?'+urllib.parse.urlencode(params)
(p/'hf_overlap_query.json').write_text(json.dumps(dict(url=url,**params),indent=2))
try:
    with urllib.request.urlopen(url,timeout=60) as response:
        data=response.read()
    (p/'hf_overlap_response.json').write_bytes(data)
    result=json.loads(data)
    print({k:v for k,v in result.items() if k not in ['features','rows']})
    print('returned_rows',len(result.get('rows',[])))
except Exception as exc:
    (p/'hf_overlap_error.txt').write_text(str(exc))
    print(type(exc).__name__,str(exc))
