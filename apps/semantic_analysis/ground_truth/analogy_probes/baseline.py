import re, importlib.util
spec=importlib.util.spec_from_file_location("sa","/Users/manojj/Documents/CSE-Projects/core-barter-system/apps/semantic_analysis/main.py")
sa=importlib.util.module_from_spec(spec); spec.loader.exec_module(sa)
from sentence_transformers import SentenceTransformer, util
m=SentenceTransformer("all-MiniLM-L6-v2"); sa.model=m
def emb(t): return m.encode(t,convert_to_tensor=True)
def cos(a,b): return float(util.cos_sim(a,b).item())
def parse(p):
    topic=teacher=None;turns=[]
    for l in open(p):
        l=l.rstrip("\n")
        if l.startswith("TOPIC:"):topic=l.split(":",1)[1].strip()
        elif l.startswith("TEACHER:"):teacher=l.split(":",1)[1].strip()
        elif re.match(r"^[A-Z]: ",l):turns.append((l[0],l[3:].strip()))
    return topic,teacher,turns
def base(p,label):
    topic,teacher,turns=parse(p); T=emb(topic+". ")
    buf=[];acc=0.0;cons=0;mx=0;tot=0;inc=0;worst="none";seq=[]
    for s,t in turns:
        if s!=teacher: continue
        buf.append(t); acc+=len(t.split())*0.4
        if acc<25.0: continue
        c=cos(emb(sa.clean_text(" ".join(buf))),T); cl=sa.classify(c); tot+=1
        if cl=="incorrect": cons+=1; inc+=1
        else: cons=0
        mx=max(mx,cons)
        tier="severe" if cons>=3 else "strong" if cons==2 else "silent" if cons==1 else "-"
        for r,n in [("severe",3),("strong",2)]:
            if tier==r and ["none","silent","strong","severe"].index(r)>["none","silent","strong","severe"].index(worst): worst=r
        seq.append(f"{c:.3f}/{tier}")
        buf=[];acc=0.0
    if buf:
        c=cos(emb(sa.clean_text(" ".join(buf))),T); cl=sa.classify(c); tot+=1
        if cl=="incorrect": inc+=1
        seq.append(f"{c:.3f}(flush)")
    ot=round(100*(tot-inc)/max(tot,1),1)
    verdict="DISPUTE" if (ot<40 or worst=="severe") else ("<=PARTIAL" if ot<70 else "SUCCESSFUL-eligible")
    print(f"{label:<42} windows={tot:<3} incorrect={inc:<3} on_topic%={ot:<6} max_consec={mx}  worst={worst:<7} -> {verdict}")
    print(f"     {' '.join(seq)}")
D="/private/tmp/claude-501/-Users-manojj-Documents-CSE-Projects-core-barter-system/53481b58-b2c3-4ee8-8128-5a6153fc3662/scratchpad"
for f,l in [("case_A.txt","A  drift+return"),("case_B2_late.txt","B2 analogy, late mapping"),
            ("case_B1_incremental.txt","B1 analogy, incremental tie-backs"),("case_C_retrofit.txt","C  drift+retrofit")]:
    base(f"{D}/{f}",l)
