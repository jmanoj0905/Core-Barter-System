import re, importlib.util, sys
spec = importlib.util.spec_from_file_location("sa", "/Users/manojj/Documents/CSE-Projects/core-barter-system/apps/semantic_analysis/main.py")
sa = importlib.util.module_from_spec(spec); spec.loader.exec_module(sa)
from sentence_transformers import SentenceTransformer, util
m = SentenceTransformer("all-MiniLM-L6-v2"); sa.model = m
def emb(t): return m.encode(t, convert_to_tensor=True)
def cos(a,b): return float(util.cos_sim(a,b).item())

STOP=set("""the a an and or but if then than that this these those of to in on for with as is are was were be been being it its it's i you he she we they them his her their my your our me him us do does did doing have has had having not no nor so such very too also just about into over under again further once here there when where why how all any both each few more most other some own same can will would should could may might must shall about at by from up down out off again what which who whom because while during before after above below between through against s t re ve ll d m o y ain aren couldn didn doesn hadn hasn haven isn ma mightn mustn needn shan shouldn wasn weren won wouldn like get got go going make makes made thing things way ways lot lots still know think say said tell tells told want wants let lets one two three now next then something anything everything nothing really quite actually""".split())
def toks(t):
    return [w for w in (x.strip(".,!?;:\"'()[]{}…—-").lower() for x in t.split())
            if len(w)>=3 and w not in STOP and w.isalpha()]

def parse(path):
    topic=teacher=learner=None; turns=[]
    for line in open(path):
        line=line.rstrip("\n")
        if line.startswith("TOPIC:"): topic=line.split(":",1)[1].strip()
        elif line.startswith("TEACHER:"): teacher=line.split(":",1)[1].strip()
        elif line.startswith("LEARNER:"): learner=line.split(":",1)[1].strip()
        elif re.match(r"^[A-Z]: ", line): turns.append((line[0], line[3:].strip()))
    return topic,teacher,learner,turns

def wins(turns,teacher,thr=25.0):
    out=[];buf=[];acc=0.0
    for i,(s,t) in enumerate(turns,1):
        if s!=teacher: continue
        buf.append((i,t)); acc+=len(t.split())*0.4
        if acc>=thr: out.append(buf);buf=[];acc=0.0
    if buf: out.append(buf)
    return out

def run(path,label,digress_below=0.25):
    topic,teacher,learner,turns=parse(path)
    T=emb(topic+". ")
    ws=wins(turns,teacher)
    wtext=[sa.clean_text(" ".join(t for _,t in w)) for w in ws]
    wc=[cos(emb(x),T) for x in wtext]
    print("="*72); print(label); print("topic:",topic)
    A=set(toks(topic))          # anchor lexicon, grows with confirmed on-topic windows
    print("A_seed:",sorted(A))
    # simulate live: iterate windows in order, maintain A and (when digressing) V
    V=set(); digress=False; dig_secs=0.0; dig_windows=[]
    turn_win={}  # turn -> window index
    for k,w in enumerate(ws,1):
        for i,_ in w: turn_win[i]=k
    # anchor tokens from correct windows discovered BEFORE each point
    print("\n%-4s %-14s %-6s %-9s %s"%("W","turns","cos","state","|new V tokens|"))
    events=[]
    for k,(w,c) in enumerate(zip(ws,wc),1):
        tk=set(toks(wtext[k-1]))
        if c>=sa.UPPER:
            A|=tk
        if c<digress_below:
            if not digress: digress=True; V=set()
            newv=tk-A
            V|=newv
            dig_secs+=sum(len(t.split())*0.4 for _,t in w)
            dig_windows.append(k)
            print("%-4s %-14s %-6.3f %-9s %d  %s"%("W%d"%k,[i for i,_ in w],c,"DIGRESS",len(newv),sorted(newv)[:12]))
        else:
            print("%-4s %-14s %-6.3f %-9s"%("W%d"%k,[i for i,_ in w],c,"on-topic"))
            if digress: digress=False
    # now recompute V over the full digression span (for scoring all turns)
    dig_turns=[i for k in dig_windows for i,_ in ws[k-1]]
    Vfull=set()
    for k in dig_windows: Vfull|=set(toks(wtext[k-1]))
    Vfull-=A
    print("\ndigression windows:",dig_windows,"turns:",dig_turns,"secs=%.1f"%dig_secs)
    print("A (final, topic + confirmed-correct windows):",sorted(A))
    print("\nper-turn binding: |A hits| / |V hits|, cos(topic)")
    print("%-5s %-4s %-5s %-5s %-6s %-6s  %s"%("turn","spk","nA","nV","cosT","words","A-hits | V-hits"))
    for i,(s,t) in enumerate(turns,1):
        tt=toks(sa.clean_text(t))
        a=[x for x in tt if x in A]; v=[x for x in tt if x in Vfull]
        ct=cos(emb(sa.clean_text(t)),T)
        mark="*" if (a and v) else " "
        print("%s%-4d %-4s %-5d %-5d %-6.3f %-6d  %s | %s"%(mark,i,s,len(set(a)),len(set(v)),ct,len(t.split()),sorted(set(a)),sorted(set(v))))
    return dict(A=A,V=Vfull,T=T,ws=ws,wc=wc,wtext=wtext,turns=turns,teacher=teacher,learner=learner,dig=dig_windows)

B=run("/Users/manojj/Documents/CSE-Projects/core-barter-system/eval_dataset/scripts/person_B/sess_B08.txt","RUN2 sess_B08")
print()
C=run("/Users/manojj/Documents/CSE-Projects/core-barter-system/eval_dataset/scripts/person_C/sess_C05.txt","RUN1 sess_C05")
