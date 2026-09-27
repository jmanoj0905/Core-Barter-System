import re, importlib.util, sys
spec = importlib.util.spec_from_file_location("sa", "/Users/manojj/Documents/CSE-Projects/core-barter-system/apps/semantic_analysis/main.py")
sa = importlib.util.module_from_spec(spec); spec.loader.exec_module(sa)
from sentence_transformers import SentenceTransformer, util
m = SentenceTransformer("all-MiniLM-L6-v2"); sa.model=m
_c={}
def emb(t):
    if t not in _c: _c[t]=m.encode(t, convert_to_tensor=True)
    return _c[t]
def cos(a,b): return float(util.cos_sim(a,b).item())
STOP=set("""the a an an and or but if then than that this these those of to in on for with as is are was were be been being it its i you he she we they them his her their my your our me him us do does did doing have has had having not no nor so such very too also just about into over under again further once here there when where why how all any both each few more most other some own same can will would should could may might must shall at by from up down out off what which who whom because while during before after above below between through against like get got go going make makes made thing things way ways lot lots still know think say said tell tells told want wants let lets one two three four five six now next then something anything everything nothing really quite actually yes okay right often else ever never only always every part see back take keep kept put whole first last end point people bit much many basically another other single simply properly genuinely completely absolutely exactly whether started start starts stop stops stopped told tells until after before somebody nobody anybody everybody""".split())
def toks(t):
    return [w for w in (x.strip(".,!?;:\"'()[]{}…—-").lower() for x in t.split())
            if len(w)>=3 and w not in STOP and w.isalpha()]
def sents(t): return [x for x in re.split(r"(?<=[.!?])\s+", t) if x.strip()]
EQ=re.compile(r"\b(is|are|was|were|means|represents|becomes|corresponds|maps|equals|that's|thats)\b|'s\b",re.I)
def parse(p):
    topic=teacher=learner=None;turns=[]
    for l in open(p):
        l=l.rstrip("\n")
        if l.startswith("TOPIC:"):topic=l.split(":",1)[1].strip()
        elif l.startswith("TEACHER:"):teacher=l.split(":",1)[1].strip()
        elif l.startswith("LEARNER:"):learner=l.split(":",1)[1].strip()
        elif re.match(r"^[A-Z]: ",l):turns.append((l[0],l[3:].strip()))
    return topic,teacher,learner,turns

RHO=0.45;K=20.0;LW=0.5;LCAP=0.4
NUDGE,STRONG,SEVERE=60.0,140.0,240.0

def run(path,label,verbose=True):
    topic,teacher,learner,turns=parse(path)
    T=emb(topic+". ")
    seed=set(toks(topic)); on_cnt={};dig_cnt={};bound=set()
    def isA(w): return w in seed or (on_cnt.get(w,0)>0 and dig_cnt.get(w,0)==0)
    def isV(w): return w not in seed and dig_cnt.get(w,0)>0 and on_cnt.get(w,0)==0
    R=sa.UPPER
    debt=bond=lbond=tsec=on_sec=0.0
    sdebt=sbond=0.0                      # span-local
    buf=[];acc=0.0;wid=0;anyV=False
    rows=[];maxunb_cum=0.0;maxunb_span=0.0;first_nudge=None;tiers_cum=[];tiers_span=[]
    elapsed=0.0
    for i,(spk,txt) in enumerate(turns,1):
        clean=sa.clean_text(txt);dur=len(txt.split())*0.4;elapsed+=dur
        if anyV:
            n=0;prs=[]
            for s in sents(clean):
                st=set(toks(s));A=[x for x in st if isA(x)];V=[x for x in st if isV(x)]
                if not(A and V and EQ.search(s)):continue
                new=[(x,y) for x in A for y in V if (x,y) not in bound]
                if not new:continue
                for p in new:bound.add(p)
                n+=min(len(A),len(V));prs.append((A[:4],V[:4]))
            if n:
                add=K*n*(LW if spk==learner else 1.0)
                if spk==learner:
                    al=max(0.0,LCAP*(bond+add)-lbond);add=min(add,al);lbond+=add
                bond+=add;sbond+=add
                if verbose and add>0:
                    print(f"      t{i:<3} {'LEARNER' if spk==learner else 'teacher'} bind x{n} +{add:.0f}s  {prs}")
        if spk!=teacher: continue
        tsec+=dur;buf.append((i,txt));acc+=dur
        if acc<25.0: continue
        wid+=1;wtxt=sa.clean_text(" ".join(t for _,t in buf));wsec=acc
        c=cos(emb(wtxt),T);thr=RHO*R;wt=set(toks(wtxt))
        if c<thr:
            debt+=wsec;sdebt+=wsec
            for w in wt: dig_cnt[w]=dig_cnt.get(w,0)+1
            anyV=True;st="DIGRESS"
        else:
            on_sec+=wsec
            for w in wt: on_cnt[w]=on_cnt.get(w,0)+1
            R=max(R,min(c,0.80));st="on-topic";sdebt=0.0;sbond=0.0
        uc=max(0.0,debt-bond);us=max(0.0,sdebt-sbond)
        maxunb_cum=max(maxunb_cum,uc);maxunb_span=max(maxunb_span,us)
        tc="SEVERE" if uc>=SEVERE else "STRONG" if uc>=STRONG else "NUDGE" if uc>=NUDGE else "silent" if uc>0 else "-"
        ts="SEVERE" if us>=SEVERE else "STRONG" if us>=STRONG else "NUDGE" if us>=NUDGE else "silent" if us>0 else "-"
        if ts=="NUDGE" and first_nudge is None: first_nudge=(wid,round(elapsed))
        tiers_cum.append(tc);tiers_span.append(ts)
        rows.append((wid,[i for i,_ in buf],c,thr,st,wsec,debt,bond,uc,tc,us,ts))
        buf=[];acc=0.0
    if buf:
        wid+=1;wtxt=sa.clean_text(" ".join(t for _,t in buf));c=cos(emb(wtxt),T);thr=RHO*R
        if c<thr: debt+=acc;sdebt+=acc;st="DIGRESS"
        else: on_sec+=acc;st="on-topic"
        uc=max(0.0,debt-bond);us=max(0.0,sdebt-sbond)
        rows.append((wid,[i for i,_ in buf],c,thr,st,acc,debt,bond,uc,"(flush)",us,"(flush)"))
    print("="*104);print(label,"|",topic)
    print(f"{'W':<4}{'turns':<10}{'cos':>7}{'thr':>7}  {'state':<8}{'sec':>6}{'debt':>8}{'bond':>8}{'unb_cum':>9} {'tier_cum':<9}{'unb_span':>9} {'tier_span':<9}")
    for (w,t,c,thr,st,sec,d,b,uc,tc,us,ts) in rows:
        print(f"W{w:<3}{str(t):<10}{c:7.3f}{thr:7.3f}  {st:<8}{sec:6.1f}{d:8.1f}{b:8.1f}{uc:9.1f} {tc:<9}{us:9.1f} {ts:<9}")
    mt=on_sec/max(tsec,1);me=on_sec/max(elapsed,1)
    print(f"  teacher={tsec:.1f}s  elapsed={elapsed:.1f}s  on_topic={on_sec:.1f}s  debt={debt:.1f}s  bond={bond:.1f}s  unbonded={max(0,debt-bond):.1f}s")
    print(f"  mass(teacher-sec)={mt:.0%}  mass(elapsed)={me:.0%}   worst tier cum={max(tiers_cum,key=lambda x:['-','silent','NUDGE','STRONG','SEVERE'].index(x)) if tiers_cum else '-'}  worst tier span={max(tiers_span,key=lambda x:['-','silent','NUDGE','STRONG','SEVERE'].index(x)) if tiers_span else '-'}")
    print(f"  first NUDGE (span ladder): {first_nudge}")
    return dict(mt=mt,me=me,debt=debt,bond=bond,unb=max(0,debt-bond))

base="/Users/manojj/Documents/CSE-Projects/core-barter-system/eval_dataset/scripts"
D="/private/tmp/claude-501/-Users-manojj-Documents-CSE-Projects-core-barter-system/53481b58-b2c3-4ee8-8128-5a6153fc3662/scratchpad"
for p,l in [(f"{D}/case_A.txt","CASE A  drift 10 windows, then genuine return"),
            (f"{D}/case_B2_late.txt","CASE B2 10-window analogy, mapping ONLY at the end"),
            (f"{D}/case_B1_incremental.txt","CASE B1 10-window analogy, incremental tie-backs + end mapping"),
            (f"{D}/case_C_retrofit.txt","CASE C  drift 10 windows + cheap late retrofit"),
            (f"{base}/person_B/sess_B08.txt","REF    sess_B08"),
            (f"{base}/person_C/sess_C05.txt","REF    sess_C05")]:
    run(p,l);print()
