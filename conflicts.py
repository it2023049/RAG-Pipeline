"""Instructions for reporting conflicting evidence in the generated answer."""

# This text is added to the system prompt in generation.py. It asks the LLM to
# acknowledge contradictions in retrieved passages. It does not independently
# detect conflicts, rank source reliability, or decide which source is correct
CONFLICT_INSTRUCTION = """
If the passages give conflicting information about the same fact,
clearly state that the evidence conflicts and cite the relevant sources.
Do not arbitrarily choose one source as correct.
If different answers refer to different entities,
explain which answer belongs to each entity.
"""
