"""Fetch India.zip from official Figshare RDD2022 v1 without downloading all 13 GB.
Pinned outer ZIP file 38030910 stores the India entry uncompressed.
"""
import argparse,hashlib,json,struct,urllib.request,zipfile,zlib
from pathlib import Path
URL='https://ndownloader.figshare.com/files/38030910'
HEADER=609341032;SIZE=526709145;TOTAL=13264172619;CRC=0xf186f6f8

def download(output):
    out=Path(output);out.parent.mkdir(parents=True,exist_ok=True)
    if out.exists():raise ValueError('Output already exists; use it or choose a new path')
    with urllib.request.urlopen(urllib.request.Request(URL,headers={'Range':f'bytes={HEADER}-{HEADER+46}'}),timeout=60) as r:
        if r.status!=206 or r.headers.get('Content-Range')!=f'bytes {HEADER}-{HEADER+46}/{TOTAL}':raise ValueError('Header range not honored')
        h=r.read(48)
    if len(h)!=47 or h[:4]!=b'PK\x03\x04' or h[30:]!=b'RDD2022/India.zip':raise ValueError('Upstream archive layout changed')
    fields=struct.unpack('<4s5H3I2H',h[:30])
    if fields[3]!=0 or fields[6]!=CRC or fields[7]!=SIZE or fields[9:]!=(17,0):raise ValueError('Unexpected archive entry')
    start=HEADER+47;end=start+SIZE-1;temp=out.with_suffix('.partial');crc=0;total=0;sha=hashlib.sha256()
    try:
        with urllib.request.urlopen(urllib.request.Request(URL,headers={'Range':f'bytes={start}-{end}'}),timeout=90) as r,temp.open('wb') as f:
            if r.status!=206 or r.headers.get('Content-Range')!=f'bytes {start}-{end}/{TOTAL}':raise ValueError('Country range not honored; refusing full 13 GB download')
            while block:=r.read(1024*1024):
                total+=len(block)
                if total>SIZE:raise ValueError('Response too large')
                crc=zlib.crc32(block,crc);sha.update(block);f.write(block)
                if total%(32*1024*1024)<1024*1024:print(f'Downloaded {total//1048576} / {SIZE//1048576} MiB',flush=True)
        if total!=SIZE or crc!=CRC:raise ValueError('Archive size/CRC mismatch')
        with zipfile.ZipFile(temp) as z:
            if z.testzip() is not None:raise ValueError('Corrupt country archive')
        temp.replace(out)
    except BaseException:
        temp.unlink(missing_ok=True);raise
    provenance={'dataset':'RDD2022 India','source_url':URL,'figshare_article':'https://doi.org/10.6084/m9.figshare.21431547.v1','outer_member':'RDD2022/India.zip','bytes':total,'sha256':sha.hexdigest(),'crc32':f'{crc:08x}','verification':'Country size/CRC and all inner member CRCs verified; full outer archive MD5 not checked.'}
    out.with_suffix('.provenance.json').write_text(json.dumps(provenance,indent=2));print(json.dumps(provenance,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',default='data/downloads/RDD2022_India.zip');a=p.parse_args();download(a.output)
