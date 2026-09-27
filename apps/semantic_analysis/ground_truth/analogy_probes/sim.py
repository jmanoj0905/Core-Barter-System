import re, importlib.util, math, sys
spec = importlib.util.spec_from_file_location("sa", "/Users/manojj/Documents/CSE-Projects/core-barter-system/apps/semantic_analysis/main.py")
sa = importlib.util.module_from_spec(spec); spec.loader.exec_module(sa)
from sentence_transformers import SentenceTransformer, util
m = SentenceTransformer("all-MiniLM-L6-v2"); sa.model = m
_c={}
def emb(t):
    if t not in _c: _c[t]=m.encode(t, convert_to_tensor=True)
    return _c[t]
def cos(a,b): return float(util.cos_sim(a,b).item())

STOP=set("""the a an and or but if then than that this these those of to in on for with as is are was were be been being it its i you he she we they them his her their my your our me him us do does did doing have has had having not no nor so such very too also just about into over under again further once here there when where why how all any both each few more most other some own same can will would should could may might must shall at by from up down out off what which who whom because while during before after above below between through against like get got go going make makes made thing things way ways lot lots still know think say said tell tells told want wants let lets one two three now next then something anything everything nothing really quite actually yes okay right often else ever never only always every part see back take keep kept put""".split())
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

# ---- token distinctiveness probe -------------------------------------------
def distinct_probe(topic, words):
    T=emb(topic+". ")
    return sorted(((cos(emb(w),T),w) for w in words), reverse=True)

# ---- the simulation --------------------------------------------------------
DIG=0.25           # window cos below this opens/extends a digression
TAU=0.18           # token must have cos(token,topic) >= TAU to count as an anchor token
K=18.0             # seconds of bond posted per distinct binding pair
LEARNER_W=0.5      # learner bindings discounted
LEARNER_CAP=0.4    # learner bindings may fund at most this share of total bond
CAP_FRAC=0.45      # hard ceiling: vehicle seconds / teacher seconds
NUDGE, STRONG, SEVERE = 60.0, 125.0, 200.0

def simulate(path, label, verbose=True):
    topic,teacher,learner,turns=parse(path)
    T=emb(topic+". ")
    A=set(); Ares={}
    def anchorable(w):
        if w in Ares: return Ares[w]
        Ares[w]= cos(emb(w),T)>=TAU
        return Ares[w]
    for w in toks(topic):
        if True: A.add(w)          # topic-string tokens are anchors by definition
    V=set()
    debt=0.0; bond=0.0; lbond=0.0; teacher_secs=0.0
    buf=[]; acc=0.0; wid=0; open_dig=False
    events=[]; log=[]
    print("="*74); print(label, "|", topic)
    print("anchor seed:", sorted(A))
    def bindings(text, spk):
        n=0; pairs=[]
        for s in sents(text):
            st=set(toks(s))
            a=sorted(x for x in st if x in A); v=sorted(x for x in st if x in V)
            if a and v:
                n+=min(len(a),len(v)); pairs.append((a,v))
        return n, pairs
    for i,(spk,txt) in enumerate(turns,1):
        clean=sa.clean_text(txt); dur=len(txt.split())*0.4
        # --- binding check on every utterance, before it changes state
        if open_dig or V:
            n,pairs = bindings(clean, spk)
            if n:
                add = K*n*(LEARNER_W if spk==learner else 1.0)
                if spk==learner:
                    allowed=max(0.0, LEARNER_CAP*(bond+add)-lbond)
                    add=min(add,allowed); lbond+=add
                bond+=add
                events.append((i,spk,n,add,pairs))
                if verbose: print(f"  t{i:<3} {'LEARNER' if spk==learner else 'teacher'} BIND x{n} +{add:.0f}s bond  {pairs}")
        if spk!=teacher:
            continue
        teacher_secs+=dur; buf.append((i,txt)); acc+=dur
        if acc<25.0: continue
        wid+=1; wtxt=sa.clean_text(" ".join(t for _,t in buf)); wsec=acc
        c=cos(emb(wtxt),T); cls=sa.classify(c)
        wt=set(toks(wtxt))
        if c<DIG:
            open_dig=True
            debt+=wsec
            V |= {w for w in wt if w not in A}
        else:
            open_dig=False
            if c>=sa.UPPER:
                A |= {w for w in wt if anchorable(w)}
        unb=max(0.0, debt-bond)
        capped = teacher_secs>0 and (debt/teacher_secs)>CAP_FRAC and debt>SEVERE
        act="none"
        if capped: act="SEVERE(cap)"
        elif unb>=SEVERE: act="SEVERE"
        elif unb>=STRONG: act="STRONG"
        elif unb>=NUDGE: act="NUDGE"
        elif unb>0: act="silent"
        log.append((wid,[i for i,_ in buf],round(c,3),cls,round(wsec,1),round(debt),round(bond),round(unb),act))
        print(f"  W{wid:<2} turns={str([i for i,_ in buf]):<10} cos={c:.3f} {cls:<15} sec={wsec:5.1f} debt={debt:6.1f} bond={bond:6.1f} unbonded={unb:6.1f}  -> {act}")
        buf=[]; acc=0.0
    if buf:
        wid+=1; wtxt=sa.clean_text(" ".join(t for _,t in buf)); c=cos(emb(wtxt),T)
        if c<DIG: debt+=acc
        print(f"  W{wid:<2} (flush) turns={[i for i,_ in buf]} cos={c:.3f}")
    print(f"  FINAL debt={debt:.1f}s bond={bond:.1f}s unbonded={max(0,debt-bond):.1f}s teacher={teacher_secs:.1f}s  vehicle_frac={debt/max(teacher_secs,1):.2f}")
    return dict(A=A,V=V,debt=debt,bond=bond,events=events,log=log)

print(">>> token distinctiveness probe (cos(word, topic)) for sess_B08")
for s,w in distinct_probe("Machine learning intro (training data, features, overfitting, validation)",
    ["data","features","model","training","validation","overfitting","dataset","label","production","lab",
     "right","often","take","small","look","turn","today","nobody","curve","error","regularisation","pattern",
     "memorise","leakage","mentor","cellar","wine","decanter","pours","apprentice","waiter","kitchen"]):
    print(f"   {s:+.3f}  {w}")
print()
print(">>> token distinctiveness probe for sess_C05")
for s,w in distinct_probe("Node.js basics (event loop, non-blocking I/O, callbacks)",
    ["thread","callback","callbacks","queue","event","loop","javascript","node","async","blocking",
     "settimeout","microtasks","promise","waiter","kitchen","ticket","rail","stove","plate","bell","sauce","restaurant",
     "right","often","small","back","picture"]):
    print(f"   {s:+.3f}  {w}")
print()
B=simulate("/Users/manojj/Documents/CSE-Projects/core-barter-system/eval_dataset/scripts/person_B/sess_B08.txt","RUN 2  sess_B08")
print()
C=simulate("/Users/manojj/Documents/CSE-Projects/core-barter-system/eval_dataset/scripts/person_C/sess_C05.txt","RUN 1  sess_C05")
