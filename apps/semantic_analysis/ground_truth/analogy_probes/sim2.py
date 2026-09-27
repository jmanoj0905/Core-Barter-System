import re, importlib.util
spec = importlib.util.spec_from_file_location("sa", "/Users/manojj/Documents/CSE-Projects/core-barter-system/apps/semantic_analysis/main.py")
sa = importlib.util.module_from_spec(spec); spec.loader.exec_module(sa)
from sentence_transformers import SentenceTransformer, util
m = SentenceTransformer("all-MiniLM-L6-v2"); sa.model = m
_c={}
def emb(t):
    if t not in _c: _c[t]=m.encode(t, convert_to_tensor=True)
    return _c[t]
def cos(a,b): return float(util.cos_sim(a,b).item())
STOP=set("""the a an and or but if then than that this these those of to in on for with as is are was were be been being it its i you he she we they them his her their my your our me him us do does did doing have has had having not no nor so such very too also just about into over under again further once here there when where why how all any both each few more most other some own same can will would should could may might must shall at by from up down out off what which who whom because while during before after above below between through against like get got go going make makes made thing things way ways lot lots still know think say said tell tells told want wants let lets one two three now next then something anything everything nothing really quite actually yes okay right often else ever never only always every part see back take keep kept put whole first last end thing point people something bit much many also basically
""".split())
def toks(t):
    return [w for w in (x.strip(".,!?;:\"'()[]{}…—-").lower() for x in t.split())
            if len(w)>=3 and w not in STOP and w.isalpha()]
def sents(t): return [s for s in re.split(r"(?<=[.!?])\s+", t) if s.strip()]
def parse(path):
    topic=teacher=learner=None; turns=[]
    for line in open(path):
        line=line.rstrip("\n")
        if line.startswith("TOPIC:"): topic=line.split(":",1)[1].strip()
        elif line.startswith("TEACHER:"): teacher=line.split(":",1)[1].strip()
        elif line.startswith("LEARNER:"): learner=line.split(":",1)[1].strip()
        elif re.match(r"^[A-Z]: ", line): turns.append((line[0], line[3:].strip()))
    return topic,teacher,learner,turns

RHO=0.45; K=20.0; LW=0.5; LCAP=0.4
NUDGE,STRONG,SEVERE = 60.0,140.0,240.0
MASS_MIN=0.50   # end-of-session: on-topic teacher seconds / total teacher seconds

def run(path,label):
    topic,teacher,learner,turns=parse(path)
    T=emb(topic+". ")
    A=set(toks(topic)); V=set(); bound=set()
    R=sa.UPPER
    debt=0.0; bond=0.0; lbond=0.0; tsec=0.0; ontopic_sec=0.0
    buf=[];acc=0.0;wid=0; span=0.0; maxspan=0.0
    print("="*76); print(label,"|",topic); print("  anchor seed:",sorted(A))
    def binds(text,spk):
        n=0;pairs=[]
        for s in sents(text):
            st=set(toks(s)); a=sorted(x for x in st if x in A); v=sorted(x for x in st if x in V)
            new=[(x,y) for x in a for y in v if (x,y) not in bound]
            if a and v and new:
                for p in new: bound.add(p)
                k=min(len(a),len(v)); n+=k; pairs.append((a,v))
        return n,pairs
    for i,(spk,txt) in enumerate(turns,1):
        clean=sa.clean_text(txt); dur=len(txt.split())*0.4
        if V:
            n,p=binds(clean,spk)
            if n:
                add=K*n*(LW if spk==learner else 1.0)
                if spk==learner:
                    allowed=max(0.0,LCAP*(bond+add)-lbond); add=min(add,allowed); lbond+=add
                bond+=add
                if add>0: print(f"    t{i:<3} {'LEARNER':<7} BIND x{n} +{add:.0f}s  {p}")
        if spk!=teacher: continue
        tsec+=dur; buf.append((i,txt)); acc+=dur
        if acc<25.0: continue
        wid+=1; wtxt=sa.clean_text(" ".join(t for _,t in buf)); wsec=acc
        c=cos(emb(wtxt),T); thr=RHO*R
        wt=set(toks(wtxt))
        if c<thr:
            debt+=wsec; span+=wsec; maxspan=max(maxspan,span)
            V |= (wt-A)
            state="DIGRESS"
        else:
            ontopic_sec+=wsec; span=0.0
            A |= wt; R=max(R,min(c,0.80)); state="on-topic"
        unb=max(0.0,debt-bond)
        act = "SEVERE" if unb>=SEVERE else "STRONG" if unb>=STRONG else "NUDGE" if unb>=NUDGE else "silent" if unb>0 else "none"
        print(f"  W{wid:<2} t={str([i for i,_ in buf]):<10} cos={c:.3f} thr={thr:.3f} {state:<8} sec={wsec:5.1f} debt={debt:6.1f} bond={bond:6.1f} unb={unb:6.1f} -> {act}")
        buf=[];acc=0.0
    if buf:
        wid+=1; wtxt=sa.clean_text(" ".join(t for _,t in buf)); c=cos(emb(wtxt),T); thr=RHO*R
        tsec_add=acc
        if c<thr: debt+=acc; span+=acc; maxspan=max(maxspan,span); st="DIGRESS"
        else: ontopic_sec+=acc; st="on-topic"
        print(f"  W{wid:<2} t={str([i for i,_ in buf]):<10} cos={c:.3f} thr={thr:.3f} {st:<8} (flush)")
    mass=ontopic_sec/max(tsec,1)
    print(f"  --- teacher={tsec:.1f}s on-topic_mass={ontopic_sec:.1f}s ({mass:.0%})  debt={debt:.1f}s  bond={bond:.1f}s unbonded={max(0,debt-bond):.1f}s  longest_span={maxspan:.1f}s")
    print(f"  SETTLEMENT: mass {mass:.0%} vs min {MASS_MIN:.0%} -> {'PASS' if mass>=MASS_MIN else 'FAIL (veto)'}")
import sys
base="/Users/manojj/Documents/CSE-Projects/core-barter-system/eval_dataset/scripts"
D="/private/tmp/claude-501/-Users-manojj-Documents-CSE-Projects-core-barter-system/53481b58-b2c3-4ee8-8128-5a6153fc3662/scratchpad"
run(f"{base}/person_C/sess_C05.txt","RUN 1  sess_C05 (2-window analogy)")
print()
run(f"{base}/person_B/sess_B08.txt","RUN 2  sess_B08 (5-window analogy)")
print()
run(f"{D}/adv_never_return.txt","ADV-1  announce analogy, never return")
print()
run(f"{D}/adv_collude.txt","ADV-2  colluding learner, one bridge per window")
