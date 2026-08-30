import json, numpy as np, cv2, random, sys
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
S="/tmp/claude-1000/-home-b920405-git-BiternionNet/959a94f5-0873-4077-ba6e-b5566517c153/scratchpad"
H="/home/b920405/git/High-Angle_Robust_Fast_FaceAlignment/data"
r=[json.loads(l) for l in open(f"{S}/sixd_synthetic.jsonl")]
for x in r: x["sixd_yaw"] = -x["sixd_yaw"]   # SixDRepNet yaw sign is opposite to the generation-intent / pan convention
sy=np.array([x["sixd_yaw"] for x in r]); iy=np.array([x["yaw_intent"] for x in r]); sp=np.array([x["sixd_pitch"] for x in r]); ip=np.array([x["pitch_intent"] for x in r]); cam=np.array([x["cam_intent"] for x in r])
print("n =", len(r))
print("sixd yaw range %.0f..%.0f ; |sixd_yaw|>90: %d ; direction=back?: %d" % (sy.min(), sy.max(), (abs(sy)>90).sum(), sum(1 for x in r if x["direction"] and "back" in x["direction"] and "front" not in x["direction"])))
# sign agreement between sixd and intent for |intent|>=30
m=abs(iy)>=30; agree=(np.sign(sy[m])==np.sign(iy[m])).mean(); print("sign agreement sixd vs intent (|intent|>=30): %.0f%%" % (100*agree))
d=sy-iy; print("sixd - intent yaw: median |d| %.1f, 90%% %.1f (all); for |intent|>=45: median |d| %.1f" % (np.median(abs(d)), np.percentile(abs(d),90), np.median(abs(d[abs(iy)>=45]))))
# usable set: sixd and intent agree within 25 deg, |sixd_pitch|<=35 (near-level head), cam in {0} or camera_high
usable=[x for x in r if abs(x["sixd_yaw"]-x["yaw_intent"])<=25 and abs(x["sixd_pitch"])<=35]
print("usable (|sixd-intent|<=25, |sixd_pitch|<=35):", len(usable))
pan=np.array([x["sixd_yaw"] % 360 for x in usable])   # pan = +yaw (0 = facing camera), sign from montage
tc=[json.loads(l) for l in open("/home/b920405/git/BiternionNet/data/towncentre/manifest.jsonl")]
tcp=np.array([x["angle_deg"] for x in tc if x["split"]=="train"])
def hist(a, w=10):
    n=int(360/w); c,_=np.histogram((a+w/2)%360, bins=np.linspace(0,360,n+1)); return np.arange(n)*w, c
cen, ctc = hist(tcp); _, csy = hist(pan)
# effective after flip
ctc_flip = (ctc + ctc[(-np.arange(36))%36]) / 2
print("\n10-deg bins where synthetic would add >= 20% of TownCentre train count:")
for c_, a, b, f in zip(cen, ctc, csy, ctc_flip):
    if b>0 and b>=0.2*a: print(f"  pan {c_:>3}: TC train {a:>4} (flip-effective {f:6.1f})  + synthetic {b:>4}")
print("\n45-deg bins: TC train / flip-effective / synthetic usable")
for c_ in range(0,360,45):
    m1=abs((tcp-c_+180)%360-180)<=22.5; m2=abs((pan-c_+180)%360-180)<=22.5
    m1f=abs((-tcp-c_+180)%360-180)<=22.5
    print(f"  {c_:>3}: {m1.sum():>4} / {(m1.sum()+m1f.sum())/2:6.1f} / {m2.sum():>4}")
# chart
fig,ax=plt.subplots(figsize=(11,4), facecolor="#fcfcfb"); ax.set_facecolor("#fcfcfb")
ax.bar(cen, ctc, width=8.6, color="#2a78d6", label="TownCentre train (real)", zorder=3)
ax.bar(cen, csy, width=8.6, bottom=ctc, color="#eb6834", label="synthetic (SixDRepNet yaw -> pan, filtered)", zorder=3)
ax.set_xticks(range(0,360,45)); ax.set_xticklabels([f"{t}°" for t in range(0,360,45)]); ax.grid(axis="y", color="#e4e3df", zorder=0)
for s in ("top","right","left"): ax.spines[s].set_visible(False)
ax.set_title("What the HRFFA synthetic heads could add, per 10° pan bin", loc="left", fontweight="bold")
ax.legend(frameon=False); ax.set_ylabel("heads"); ax.set_xlabel("pan (0° = facing camera)")
fig.tight_layout(); fig.savefig(f"{S}/synthetic_fill.jpg", dpi=150, facecolor="#fcfcfb")
# montage of usable candidates in the sparse region 20-80 deg and 280-340, downscaled to 28px then shown at 92px
random.seed(0)
def row(lo,hi):
    sel=[x for x in usable if lo<=x["sixd_yaw"]%360<=hi]; sel=random.sample(sel, min(10,len(sel))); tiles=[]
    for x in sel:
        im=cv2.imread(f"{H}/{x['dataset']}/images/{x['filename']}"); x0,y0,x1,y1=[int(v) for v in x["head_box"]]
        cx,cy=(x0+x1)//2,(y0+y1)//2; s=int(max(x1-x0,y1-y0)*1.15)//2; crop=im[max(0,cy-s):cy+s,max(0,cx-s):cx+s]
        t=cv2.resize(cv2.resize(crop,(28,28),interpolation=cv2.INTER_AREA),(92,92),interpolation=cv2.INTER_NEAREST)
        cv2.putText(t,f"{int(x['sixd_yaw']%360)}",(2,12),cv2.FONT_HERSHEY_SIMPLEX,0.4,(0,255,255),1); tiles.append(t)
    return np.hstack(tiles) if tiles else np.zeros((92,920,3),np.uint8)
cv2.imwrite(f"{S}/synthetic_candidates.png", np.vstack([row(20,80), row(280,340)]))
print("wrote synthetic_fill.jpg, synthetic_candidates.png (rows: pan 20-80 / 280-340)")
