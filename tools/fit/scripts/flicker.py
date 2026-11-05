# 闪烁尖峰统计（总负责人 flicker.py 的分镜头版）：我们的帧间变化 > 3 × 原视频且 > 4（灰度，320×180，角色遮罩内）
#   tools/.venv/bin/python -I tools/fit/scripts/flicker.py <渲染目录> <起帧-止帧(含)>
import sys
import cv2, numpy as np, subprocess, json, os, bisect
R=sys.argv[1]; FA,FB=map(int,sys.argv[2].split('-'))
W,H=320,180
p=subprocess.Popen(['ffmpeg','-v','error','-i','assets/video/DeepSeek-chan.mp4','-vf',f'scale={W}:{H}','-f','rawvideo','-pix_fmt','gray','-'],stdout=subprocess.PIPE)
prev_ref=None; prev_r=None; prev_a=None; out=[]
sh=json.load(open('scenes/deepseek-chan/shots.json'))['shots']
starts=set(s['start'] for s in sh)
for f in range(FB+1):
    ref=np.frombuffer(p.stdout.read(W*H),np.uint8).astype(np.float32).reshape(H,W)
    fn=f'{R}/frame_{f:06d}.png'
    if f < FA - 1: prev_ref=ref; continue
    a=g=None
    if os.path.exists(fn):
        im=cv2.imread(fn,cv2.IMREAD_UNCHANGED); im=cv2.resize(im,(W,H),interpolation=cv2.INTER_AREA)
        a=im[:,:,3].astype(np.float32)/255; g=cv2.cvtColor(im[:,:,:3],cv2.COLOR_BGR2GRAY).astype(np.float32)
    if a is not None and prev_a is not None and f not in starts:
        m=(a>0.5)&(prev_a>0.5)
        if m.sum()>200:
            out.append((f,float(np.abs(g-prev_r)[m].mean()),float(np.abs(ref-prev_ref)[m].mean()), float(np.abs(a-prev_a).mean()*100)))
    prev_ref=ref; prev_r=g; prev_a=a
st=[s['start'] for s in sh]; ids=[s['id'] for s in sh]
spk=[x for x in out if x[1]>3*max(x[2],1.0) and x[1]>4]
by={}
for f,*_ in spk: by.setdefault(ids[bisect.bisect_right(st,f)-1],[]).append(f)
print('样本帧',len(out),'抖动尖峰帧',len(spk))
for k,v in sorted(by.items(),key=lambda x:-len(x[1])): print(' ',k,len(v),v[:10])
dr=np.array([x[1] for x in out]); df=np.array([x[2] for x in out])
print('帧间变化中位：渲染',round(float(np.median(dr)),2),'原视频',round(float(np.median(df)),2))
# 交替抖动：相邻差分符号模式（渲染变化大而原视频 0，即原视频重复帧时我们在动）
dup=[x for x in out if x[2]<0.3 and x[1]>2]
by2={}
for f,*_ in dup: by2.setdefault(ids[bisect.bisect_right(st,f)-1],0); by2[ids[bisect.bisect_right(st,f)-1]]+=1
print('原视频静止(重复帧)但我们在变的帧',len(dup), dict(sorted(by2.items(),key=lambda x:-x[1])[:12]))
p.kill()
