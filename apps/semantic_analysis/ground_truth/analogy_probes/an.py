import sys, re, itertools
sys.path.insert(0, "/Users/manojj/Documents/CSE-Projects/core-barter-system/apps/semantic_analysis")
import importlib.util
spec = importlib.util.spec_from_file_location("sa", "/Users/manojj/Documents/CSE-Projects/core-barter-system/apps/semantic_analysis/main.py")
sa = importlib.util.module_from_spec(spec); spec.loader.exec_module(sa)
from sentence_transformers import SentenceTransformer, util
m = SentenceTransformer("all-MiniLM-L6-v2")
sa.model = m
import torch
def emb(t): return m.encode(t, convert_to_tensor=True)
def cos(a,b): return float(util.cos_sim(a,b).item())

def parse(path):
    topic=None; teacher=None; learner=None; turns=[]
    for line in open(path):
        line=line.rstrip("\n")
        if line.startswith("TOPIC:"): topic=line.split(":",1)[1].strip()
        elif line.startswith("TEACHER:"): teacher=line.split(":",1)[1].strip()
        elif line.startswith("LEARNER:"): learner=line.split(":",1)[1].strip()
        elif re.match(r"^[A-Z]: ", line):
            spk=line[0]; txt=line[3:].strip(); turns.append((spk,txt))
    return topic,teacher,learner,turns

def windows(turns, teacher, thr=25.0):
    """replicate buffering: teacher segments only, dur = words*0.4"""
    out=[]; buf=[]; acc=0.0
    for i,(spk,txt) in enumerate(turns, start=1):
        if spk!=teacher: continue
        d=len(txt.split())*0.4
        buf.append((i,txt)); acc+=d
        if acc>=thr:
            out.append(buf); buf=[]; acc=0.0
    if buf: out.append(buf)
    return out

def report(path, label):
    topic,teacher,learner,turns=parse(path)
    T=emb(topic+". ")
    print("="*70); print(label, "TOPIC:", topic)
    ws=windows(turns,teacher)
    wtexts=[sa.clean_text(" ".join(t for _,t in w)) for w in ws]
    wemb=[emb(t) for t in wtexts]
    wcos=[cos(e,T) for e in wemb]
    print("\n-- windows: turns, cos(topic), classify")
    for k,(w,c) in enumerate(zip(ws,wcos),1):
        print(f"  W{k:<2} turns={[i for i,_ in w]} cos={c:.3f} {sa.classify(c)}")
    print("\n-- window-to-window cosine (adjacent)")
    for k in range(1,len(wemb)):
        print(f"  W{k}->W{k+1}: {cos(wemb[k-1],wemb[k]):.3f}")
    print("\n-- full window x window matrix")
    hdr="      "+"".join(f"W{j+1:<5}" for j in range(len(wemb)))
    print(hdr)
    for i in range(len(wemb)):
        print(f"  W{i+1:<3} "+"".join(f"{cos(wemb[i],wemb[j]):<6.3f}" for j in range(len(wemb))))
    # learner turns
    print("\n-- learner turns: cos(topic)")
    lt=[(i,txt) for i,(spk,txt) in enumerate(turns,start=1) if spk==learner]
    lemb={i:emb(sa.clean_text(txt)) for i,txt in lt}
    for i,txt in lt:
        print(f"  t{i:<3} cosT={cos(lemb[i],T):.3f}  \"{txt[:70]}\"")
    return dict(topic=topic,teacher=teacher,learner=learner,turns=turns,T=T,ws=ws,
                wtexts=wtexts,wemb=wemb,wcos=wcos,lt=lt,lemb=lemb,emb=emb,cos=cos)

B=report("/Users/manojj/Documents/CSE-Projects/core-barter-system/eval_dataset/scripts/person_B/sess_B08.txt","RUN2 B08")
C=report("/Users/manojj/Documents/CSE-Projects/core-barter-system/eval_dataset/scripts/person_C/sess_C05.txt","RUN1 C05")
import pickle
pickle.dump("done", open("/dev/null","wb"))
