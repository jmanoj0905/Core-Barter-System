"""Blinded authoring materials for the analogy corpus (spec §5.3).

This package is deliberately import-light: it holds only the prompt template
and a plain dict of behavioural category descriptions. It must never import
anything from the measurement/harness side of the codebase (no
sentence_transformers, no embed_windows, no other eval_dataset.tools module),
so that an author working purely from this package's contents has no way to
discover how the corpus will later be measured.
"""

CATEGORY_DESCRIPTIONS: dict[str, str] = {
    "gradual_drift": (
        "The teacher starts out firmly on the stated topic, then gradually "
        "stops teaching it and never comes back. The shift should not be a "
        "single abrupt jump — let it happen turn by turn, so a reader could "
        "point to the session getting further and further from the topic "
        "rather than one clean break. By the end of the session the teacher "
        "is talking about something else entirely and does not return to "
        "the topic again."
    ),
    "adversarial": (
        "For a long, sustained stretch of the session, the teacher abandons "
        "the stated topic and instead talks at length about a completely "
        "different subject — told properly, as a real subject in its own "
        "right, with the same coherence and depth the teacher would give "
        "the original topic, not as filler or nonsense. The off-topic "
        "stretch should feel deliberate and confident, as though the "
        "teacher has simply decided to teach something else for a while."
    ),
    "depressed_baseline": (
        "A calm, fully on-topic session about an abstract subject. The "
        "teacher explains the stated topic clearly and completely from "
        "start to finish and never leaves it for any stretch of the "
        "conversation — there should be no off-topic turns anywhere in "
        "this script."
    ),
    "abstract_on_topic": (
        "A sustained, unambiguously on-topic session about an abstract "
        "subject. The teacher explains the topic at length while relying "
        "mostly on general, abstract language rather than leaning heavily "
        "on the topic's own specific jargon or concrete vocabulary — the "
        "teaching should still be clearly and entirely about the stated "
        "topic throughout. The teacher must never leave the topic for any "
        "stretch of the conversation — there should be no off-topic turns "
        "anywhere in this script."
    ),
    "interleaved_abuse": (
        "The session alternates repeatedly between teaching the stated "
        "topic and talking about something else, then back to the topic, "
        "then away again, several times over the course of the "
        "conversation. Each on-topic and off-topic stretch should be long "
        "enough to feel like a real segment of conversation rather than a "
        "single stray line, and the pattern should repeat more than once."
    ),
}
