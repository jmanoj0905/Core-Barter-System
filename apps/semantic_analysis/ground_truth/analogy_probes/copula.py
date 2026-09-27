import re
EQ = re.compile(r"\b(is|are|was|were|'s|means|represents|becomes|corresponds|maps|equals|that's|thats|the same as|equivalent)\b|\bis the\b|\bare the\b", re.I)
tests = [
 ("C05 t8  TRUE ",  "Okay, and the waiter is the thread?"),
 ("C05 t9  TRUE ",  "The waiter is the single thread running your JavaScript, the kitchen is the operating system doing the file read or the network call, and the ticket rail is the queue."),
 ("C05 t10 TRUE ",  "And standing at the stove is what, a while loop?"),
 ("C05 t7  NOISE",  "The way the oak sits behind the fruit instead of sitting on top of it."),
 ("C05 t11 NOISE",  "While that runs the whole server is unresponsive, because there is only ever one waiter."),
 ("B08 t12 TRUE ",  "The morning pours are your training data."),
 ("B08 t12 TRUE ",  "The sharpness and the wet stone and where the oak sits are features."),
 ("B08 t13 TRUE ",  "And the mentor only owning two decades of one estate, that's what, bad data?"),
 ("B08 t8  NOISE",  "He never once said a word about acid levels or soil composition or harvest dates."),
 ("B08 t7  NOISE",  "Every morning the mentor pours again, and every morning he tells her what she got wrong."),
 ("B08 t26 NOISE",  "Validation error falls, bottoms out, then starts climbing."),
 ("ADV2 fake    ",  "So the fixture list is like the training data?"),
 ("ADV2 fake    ",  "Right, and Saliba being out is a bit like overfitting?"),
]
for lbl,s in tests:
    print(f"  {lbl}  copula={'YES' if EQ.search(s) else 'no ':<3}  {s[:78]}")
