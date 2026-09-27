import re, importlib.util, itertools, statistics as st
spec=importlib.util.spec_from_file_location("sa","/Users/manojj/Documents/CSE-Projects/core-barter-system/apps/semantic_analysis/main.py")
sa=importlib.util.module_from_spec(spec); spec.loader.exec_module(sa)
from sentence_transformers import SentenceTransformer, util
import torch
m=SentenceTransformer("all-MiniLM-L6-v2"); sa.model=m
_c={}
def emb(t):
    if t not in _c: _c[t]=m.encode(t,convert_to_tensor=True)
    return _c[t]
def nrm(v): return v/v.norm()
def cos(a,b): return float(util.cos_sim(a,b).item())
STOP=set(open("/dev/null").read().split()) # placeholder
STOP=set("""the a an and or but if then than that this these those of to in on for with as is are was were be been being it its i you he she we they them his her their my your our me him us do does did doing have has had having not no nor so such very too also just about into over under again further once here there when where why how all any both each few more most other some own same can will would should could may might must shall at by from up down out off what which who whom because while during before after above below between through against like get got go going make makes made thing things way ways lot lots still know think say said tell tells told want wants let lets one two three four five six seven eight nine ten now next then something anything everything nothing really quite actually yes okay right often else ever never only always every part see back take keep kept put whole first last end point people bit much many basically another single simply properly genuinely completely absolutely exactly whether started start starts stop stops stopped until after before somebody nobody anybody everybody said says going gone came come went being about half own new old good bad big small long short own thing""".split())
def toks(t):
    return [w for w in (x.strip(".,!?;:\"'()[]{}…—-").lower() for x in t.split())
            if len(w)>=3 and w not in STOP and w.isalpha()]
def parse(p):
    topic=teacher=learner=None;turns=[]
    for l in open(p):
        l=l.rstrip("\n")
        if l.startswith("TOPIC:"):topic=l.split(":",1)[1].strip()
        elif l.startswith("TEACHER:"):teacher=l.split(":",1)[1].strip()
        elif l.startswith("LEARNER:"):learner=l.split(":",1)[1].strip()
        elif re.match(r"^[A-Z]: ",l):turns.append((l[0],l[3:].strip()))
    return topic,teacher,learner,turns
RHO=0.45
def analyse(path,label,lab):
    topic,teacher,learner,turns=parse(path); T=emb(topic+". ")
    seed=set(toks(topic))
    # pass 1: build windows
    buf=[];acc=0.0;W=[]
    for i,(s,t) in enumerate(turns,1):
        if s!=teacher: continue
        buf.append((i,t)); acc+=len(t.split())*0.4
        if acc>=25.0: W.append((([i for i,_ in buf]),sa.clean_text(" ".join(x for _,x in buf)),acc)); buf=[];acc=0.0
    if buf: W.append((([i for i,_ in buf]),sa.clean_text(" ".join(x for _,x in buf)),acc))
    # pass 2: causal thr, mark digressions
    R=sa.UPPER; marks=[]
    for tl,txt,sec in W:
        c=cos(emb(txt),T); thr=RHO*R
        dig=c<thr
        if not dig: R=max(R,min(c,0.80))
        marks.append((tl,txt,sec,c,thr,dig))
    # longest contiguous digression span
    best=(0,0,0); cur=None
    for k,(_,_,_,_,_,dig) in enumerate(marks):
        if dig:
            if cur is None: cur=k
            if k-cur+1>best[0]-0: pass
        else:
            if cur is not None:
                if k-cur>best[0]: best=(k-cur,cur,k)
                cur=None
    if cur is not None and len(marks)-cur>best[0]: best=(len(marks)-cur,cur,len(marks))
    n,lo,hi=best
    span=marks[lo:hi]
    spanT=[s[1] for s in span]; spanE=[emb(x) for x in spanT]
    spanTok=[set(toks(x)) for x in spanT]
    # --- S1 mean cosine
    s1=st.mean([s[3] for s in span]) if span else 0
    # --- S2 mean adjacent window-window cosine within span
    s2=st.mean([cos(spanE[i],spanE[i+1]) for i in range(len(spanE)-1)]) if len(spanE)>1 else float('nan')
    s2all=st.mean([cos(spanE[i],spanE[j]) for i,j in itertools.combinations(range(len(spanE)),2)]) if len(spanE)>1 else float('nan')
    # --- S3 vocabulary turnover: mean Jaccard adjacent
    s3=st.mean([len(spanTok[i]&spanTok[i+1])/max(1,len(spanTok[i]|spanTok[i+1])) for i in range(len(spanTok)-1)]) if len(spanTok)>1 else float('nan')
    # --- S4 entity recurrence: tokens appearing in >=3 span windows; and recurring share
    from collections import Counter
    df=Counter()
    for s in spanTok: df.update(s)
    rec3=[w for w,c in df.items() if c>=3]
    rec2=[w for w,c in df.items() if c>=2]
    share=sum(df[w] for w in rec2)/max(1,sum(df.values()))
    # --- S5 domain clusters: single-linkage at 0.35
    k=len(spanE); par=list(range(k))
    def find(a):
        while par[a]!=a: par[a]=par[par[a]]; a=par[a]
        return a
    for i,j in itertools.combinations(range(k),2):
        if cos(spanE[i],spanE[j])>=0.35:
            a,b=find(i),find(j)
            if a!=b: par[a]=b
    clusters=len({find(i) for i in range(k)})
    # --- S6 relational isomorphism (Gentner): best alignment of difference vectors
    Ev=[w for w,_ in df.most_common(40) if w not in seed][:8]
    ontok=Counter()
    for tl,txt,sec,c,thr,dig in marks:
        if not dig: ontok.update(set(toks(txt)))
    Et=list(seed)+[w for w,_ in ontok.most_common(30) if w not in seed][:8]
    Et=Et[:12]
    def rel(a,b): return nrm(nrm(emb(a))-nrm(emb(b)))
    scores=[]
    for a,b in itertools.permutations(Ev,2):
        rv=rel(a,b); bestv=-2; bp=None
        for c_,d in itertools.permutations(Et,2):
            v=cos(rv,rel(c_,d))
            if v>bestv: bestv=v; bp=(c_,d)
        scores.append((bestv,(a,b),bp))
    scores.sort(reverse=True)
    s6=st.mean([x[0] for x in scores[:8]]) if scores else float('nan')
    # --- S7 substitution test vs append control
    subs=[]
    top3=Ev[:3]; tt=[w for w in Et][:3]
    for txt in spanT:
        sub=txt
        for a,b in zip(top3,tt):
            sub=re.sub(rf"\b{re.escape(a)}\b",b,sub,flags=re.I)
        app=txt+" "+" ".join(tt)
        subs.append(cos(emb(sub),T)-cos(emb(app),T))
    s7=st.mean(subs) if subs else float('nan')
    # --- S8 learner topic-vocabulary trajectory (unprompted anchors)
    lt=[(i,t) for i,(s,t) in enumerate(turns,1) if s==learner]
    anch=set(seed)|set(w for w,_ in ontok.most_common(40))
    rates=[]
    for i,t in lt:
        tk=toks(sa.clean_text(t))
        prev=set(toks(sa.clean_text(turns[i-2][1]))) if i>=2 else set()
        unp=[w for w in tk if w in anch and w not in prev]
        rates.append(len(unp)/max(1,len(tk)))
    h=len(rates)//2
    s8a=st.mean(rates[:h]) if h else float('nan'); s8b=st.mean(rates[h:]) if rates[h:] else float('nan')
    return dict(label=label,lab=lab,nspan=n,s1=s1,s2=s2,s2all=s2all,s3=s3,rec3=len(rec3),share=share,
                clusters=clusters,s6=s6,s7=s7,s8a=s8a,s8b=s8b,s8d=(s8b-s8a) if rates else float('nan'),
                Ev=Ev,top=scores[:4])
base="/Users/manojj/Documents/CSE-Projects/core-barter-system/eval_dataset/scripts"
D="/private/tmp/claude-501/-Users-manojj-Documents-CSE-Projects-core-barter-system/53481b58-b2c3-4ee8-8128-5a6153fc3662/scratchpad"
S=[(f"{base}/person_C/sess_C05.txt","C05 analogy 2w","ANALOGY"),
   (f"{base}/person_B/sess_B08.txt","B08 analogy 5w","ANALOGY"),
   (f"{D}/case_B1_incremental.txt","B1 analogy 10w inc","ANALOGY"),
   (f"{D}/case_B2_late.txt","B2 analogy 10w late","ANALOGY"),
   (f"{D}/case_A.txt","A  drift 10w wander","DRIFT"),
   (f"{D}/case_C_retrofit.txt","C  drift+retrofit","DRIFT"),
   (f"{D}/adv_never_return.txt","ADV1 football no ret","ADVERS"),
   (f"{D}/adv_collude.txt","ADV2 football collude","ADVERS")]
rows=[analyse(p,l,lb) for p,l,lb in S]
hdr=f"{'script':<22}{'cls':<8}{'span':>5}{'S1 cos':>8}{'S2 adj':>8}{'S2 all':>8}{'S3 jac':>8}{'S4 rec3':>8}{'share':>7}{'S5 cl':>6}{'S6 rel':>8}{'S7 sub':>8}{'S8 e':>7}{'S8 l':>7}{'S8 d':>7}"
print(hdr); print("-"*len(hdr))
for r in rows:
    print(f"{r['label']:<22}{r['lab']:<8}{r['nspan']:>5}{r['s1']:>8.3f}{r['s2']:>8.3f}{r['s2all']:>8.3f}{r['s3']:>8.3f}{r['rec3']:>8}{r['share']:>7.2f}{r['clusters']:>6}{r['s6']:>8.3f}{r['s7']:>8.3f}{r['s8a']:>7.3f}{r['s8b']:>7.3f}{r['s8d']:>+7.3f}")
print()
for r in rows:
    print(f"{r['label']:<22} vehicle entities: {r['Ev']}")
print()
for r in rows:
    print(f"{r['label']:<22} best relation alignments:")
    for sc,(a,b),bp in r['top']:
        print(f"      {sc:.3f}   {a}:{b}   ~   {bp[0]}:{bp[1]}")
