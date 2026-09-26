"""Hand-authored (topic, scope, labeled example texts) fixtures.

Each entry mimics one barter session contract: a topic+scope pair plus a
handful of example teacher utterances per label.

The authoritative definition of each label, its decision procedure and its
tie-breaks is labeling_rubric.md — read that before adding or relabeling
examples here. In short: "correct" advances something the scope clause names,
"weakly_correct" is on topic but needs one bridging sentence to reach a scope
item, and "incorrect" has no topical relation at all.

These are a synthetic stand-in for real session transcripts until enough real,
human-labeled window_results accumulate. SCOPE_COUNTERFACTUALS at the bottom of
this file is a separate diagnostic probe, not calibration data.
"""

TOPICS = [
    {
        "topic": "Python list comprehensions",
        "scope": "syntax, common patterns, and performance versus a for loop",
        "correct": [
            "A list comprehension is just a compact for loop that builds a new list in one line.",
            "You can filter items by adding an if clause at the end, like x for x in items if x > 0.",
            "Nested comprehensions let you flatten a list of lists, though it hurts readability past two levels.",
            "For large inputs a comprehension is usually faster than append in a loop because of fewer bytecode dispatches.",
        ],
        "weakly_correct": [
            "Generator expressions look almost the same but use parentheses and are lazy instead of eager.",
            "Dictionary comprehensions follow the same idea but build key-value pairs instead of a flat list.",
            "If the callback logic gets complicated, some people switch to map and filter, though comprehensions read better.",
        ],
        "incorrect": [
            "Did you catch the game last night, that overtime finish was wild.",
            "I need to pick up groceries after this, we are out of milk again.",
            "My internet has been so slow all week, I think the router needs a reset.",
        ],
    },
    {
        "topic": "Home sourdough bread baking",
        "scope": "starter maintenance, hydration ratios, and oven spring technique",
        "correct": [
            "Feed your starter a 1:1:1 ratio of starter, flour, and water every day if it lives at room temperature.",
            "A higher hydration dough is stickier to handle but gives you a more open, airy crumb.",
            "Oven spring happens in the first ten minutes, so a hot dutch oven with steam really helps the rise.",
            "Score the loaf right before baking so the crust has a controlled place to expand instead of tearing randomly.",
        ],
        "weakly_correct": [
            "You could use commercial yeast instead of a starter, it is faster but the flavor is flatter.",
            "Whole wheat flour absorbs more water than white flour, so you will need to adjust your hydration.",
            "A banneton basket helps hold the shape during the final proof, though a floured bowl works in a pinch.",
        ],
        "incorrect": [
            "I finally finished setting up my new gaming PC this weekend, the frame rates are insane.",
            "My dentist appointment got moved to next Tuesday, kind of annoying honestly.",
            "We are thinking about repainting the living room, maybe a warm gray this time.",
        ],
    },
    {
        "topic": "Beginner acoustic guitar chords",
        "scope": "open chord shapes, transitions, and basic strumming patterns",
        "correct": [
            "Start with G, C, D and E minor, those four chords cover a huge number of songs.",
            "Keep your thumb behind the neck and let your fingers curl so you are not muting the adjacent strings.",
            "Practice switching between just two chords slowly before adding a strumming pattern on top.",
            "A down-down-up-up-down-up pattern is a solid default for most folk and pop songs.",
        ],
        "weakly_correct": [
            "Barre chords let you play the same shapes higher up the neck once your hand strength improves.",
            "A capo is basically a moveable nut, it lets you keep open chord shapes in a different key.",
            "Some people learn on electric guitar first since the strings are lighter, though the technique carries over.",
        ],
        "incorrect": [
            "I am trying a new recipe for lasagna tonight, hope it turns out okay.",
            "Traffic on the highway was absolutely brutal this morning, took me an hour longer.",
            "Have you seen the new season of that show everyone is talking about?",
        ],
    },
    {
        "topic": "Filing a basic personal income tax return",
        "scope": "standard deduction, W-2 income, and common filing mistakes",
        "correct": [
            "Most people with a single W-2 and no dependents are better off taking the standard deduction.",
            "Double check that your employer's reported wages match box 1 on your W-2 before you file.",
            "Filing status changes your bracket, so married filing jointly usually has a lower effective rate than filing separately.",
            "A common mistake is forgetting to report freelance income that did not come with a 1099.",
        ],
        "weakly_correct": [
            "If you itemize instead, you would need receipts for things like mortgage interest or large medical expenses.",
            "Contributing to a traditional IRA before the deadline can still lower last year's taxable income.",
            "Self-employed folks have a different set of forms since they also owe self-employment tax.",
        ],
        "incorrect": [
            "We are planning a camping trip for next month, still deciding on the campsite.",
            "My phone battery has been draining so fast lately, might need a new one.",
            "The coffee shop down the street just started roasting their own beans, it smells amazing.",
        ],
    },
    {
        "topic": "Basic dog obedience training",
        "scope": "sit, stay, and loose-leash walking using positive reinforcement",
        "correct": [
            "Reward the sit the instant their rear touches the ground, timing matters more than the treat itself.",
            "For stay, build up duration before distance, and distance before distraction.",
            "Loose-leash walking improves fastest if you stop moving the second the leash goes tight, rather than pulling back.",
            "Short five minute sessions a few times a day beat one long exhausting session.",
        ],
        "weakly_correct": [
            "A clicker can sharpen your timing since the click always means the exact same thing to the dog.",
            "High-value treats like real chicken work better for tougher environments than plain kibble.",
            "Some trainers use a prong collar for leash pulling, though positive reinforcement usually gets there without it.",
        ],
        "incorrect": [
            "I have been meaning to reorganize my closet all week, just have not gotten around to it.",
            "The flight got delayed by three hours, we basically lived at the gate.",
            "Our office is switching to a new project management tool next month.",
        ],
    },
    {
        "topic": "Watercolor painting fundamentals",
        "scope": "wet-on-wet vs wet-on-dry technique and basic color mixing",
        "correct": [
            "Wet-on-wet gives you soft blooming edges, good for skies and backgrounds.",
            "Wet-on-dry keeps hard, controlled edges, which is what you want for sharp detail work.",
            "Mixing a color straight from the tube usually looks flatter than layering two thin transparent washes.",
            "Test your color on a scrap sheet first since watercolor dries a shade or two lighter than it looks wet.",
        ],
        "weakly_correct": [
            "Paper weight matters a lot, anything under 140lb tends to warp badly once it gets wet.",
            "A round brush is the most versatile single brush to start with before buying a whole set.",
            "Some painters use masking fluid to preserve white highlights before laying down a wash.",
        ],
        "incorrect": [
            "I switched gyms recently and the new one has way better equipment.",
            "We are trying to decide between two flights for the trip, one has a layover.",
            "My neighbor's dog keeps getting into our yard, need to fix the fence.",
        ],
    },
]


# ---------------------------------------------------------------------------
# Scope counterfactuals
# ---------------------------------------------------------------------------
#
# Random unrelated negatives (the "incorrect" lists above) are easy: small talk
# looks nothing like a lesson in any embedding space. They cannot tell us
# whether the model reads the *scope* clause it was handed, or merely recognises
# the broad topic. These pairs do.
#
# Each entry keeps one topic fixed and varies only the scope. Every utterance is
# genuine teaching content on that topic, and is labeled TWICE — once under each
# scope. A model that ignores scope scores an utterance identically under both,
# so the paired comparison is a threshold-free test: does similarity move in the
# direction the labels say it should?
#
# These rows are a DIAGNOSTIC PROBE and must never be folded into the
# calibration set. They are adversarial by construction, so fitting thresholds
# on them would bias the boundary toward hard cases that are not
# representative of ordinary session traffic. See design_decisions.md (D7).

SCOPE_COUNTERFACTUALS = [
    {
        "topic": "Python list comprehensions",
        "scope_a": "syntax, common patterns, and performance versus a for loop",
        "scope_b": "readability trade-offs and when to avoid them in team code review",
        "pairs": [
            {
                "text": "For large inputs a comprehension avoids repeated append calls, so it runs measurably faster.",
                "under_a": "correct", "under_b": "weakly_correct",
            },
            {
                "text": "Past two levels of nesting most reviewers will ask you to rewrite it as an explicit loop.",
                "under_a": "weakly_correct", "under_b": "correct",
            },
        ],
    },
    {
        "topic": "Home sourdough bread baking",
        "scope_a": "starter maintenance, hydration ratios, and oven spring technique",
        "scope_b": "choosing flour types and milling your own grain at home",
        "pairs": [
            {
                "text": "Feed the starter equal weights of flour and water once a day at room temperature.",
                "under_a": "correct", "under_b": "weakly_correct",
            },
            {
                "text": "Freshly milled whole grain turns rancid within weeks, so mill only what you will use.",
                "under_a": "weakly_correct", "under_b": "correct",
            },
        ],
    },
    {
        "topic": "Beginner acoustic guitar chords",
        "scope_a": "open chord shapes, transitions, and basic strumming patterns",
        "scope_b": "tuning the instrument and changing worn strings",
        "pairs": [
            {
                "text": "Move between G and C slowly until the change is clean, then bring the tempo up.",
                "under_a": "correct", "under_b": "weakly_correct",
            },
            {
                "text": "Stretch each new string as you bring it up to pitch or it will slip flat for days.",
                "under_a": "weakly_correct", "under_b": "correct",
            },
        ],
    },
    {
        "topic": "Filing a basic personal income tax return",
        "scope_a": "standard deduction, W-2 income, and common filing mistakes",
        "scope_b": "self-employment income and quarterly estimated payments",
        "pairs": [
            {
                "text": "If all your income sits on a W-2, the standard deduction is usually simpler than itemizing.",
                "under_a": "correct", "under_b": "weakly_correct",
            },
            {
                "text": "Send estimated tax four times a year or you will owe an underpayment penalty at filing.",
                "under_a": "weakly_correct", "under_b": "correct",
            },
        ],
    },
    {
        "topic": "Basic dog obedience training",
        "scope_a": "sit, stay, and loose-leash walking using positive reinforcement",
        "scope_b": "crate training and overnight routines for a new puppy",
        "pairs": [
            {
                "text": "Reward the instant the rear touches the floor, not after the dog stands back up.",
                "under_a": "correct", "under_b": "weakly_correct",
            },
            {
                "text": "Keep the crate beside your bed for the first few nights so the puppy settles faster.",
                "under_a": "weakly_correct", "under_b": "correct",
            },
        ],
    },
    {
        "topic": "Watercolor painting fundamentals",
        "scope_a": "wet-on-wet vs wet-on-dry technique and basic color mixing",
        "scope_b": "stretching paper and choosing brushes for a first kit",
        "pairs": [
            {
                "text": "Drop pigment into a still-wet wash and it blooms outward with soft feathered edges.",
                "under_a": "correct", "under_b": "weakly_correct",
            },
            {
                "text": "Tape damp paper down to a board or it will cockle the moment you lay a broad wash.",
                "under_a": "weakly_correct", "under_b": "correct",
            },
        ],
    },
]
