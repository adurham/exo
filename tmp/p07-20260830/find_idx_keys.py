import json, struct, glob, os
base = os.path.expanduser('~/.exo/models/deepseek-ai--DeepSeek-V4-Flash-0731')
f = base + '/model-00006-of-00048.safetensors'
with open(f, 'rb') as fh:
    n = struct.unpack('<Q', fh.read(8))[0]
    hdr = json.loads(fh.read(n))
for k, v in hdr.items():
    if 'indexer' in k:
        print(k, v['dtype'], tuple(v['shape']))