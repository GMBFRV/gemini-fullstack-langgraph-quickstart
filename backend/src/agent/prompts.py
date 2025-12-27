from datetime import datetime


def get_current_date():
    return datetime.now().strftime("%B %d, %Y")


local_reader_instructions = """You are given EXCERPTS from a LOCAL file. Write a factual summary ONLY from these excerpts.

Rules (STRICT):
- Use ONLY the provided excerpts. No outside knowledge.
- Do NOT invent external URLs or sources.
- If excerpts are empty, reply exactly: "No relevant local sources found."
- Use citations ONLY in the provided form [S#](local://...).
- Do not mention web search, Google, Wikipedia, or any external source.

Current date: {current_date}

User question:
{question}

File:
{file_path}

Excerpts:
{excerpts}
"""


reflection_instructions = """You are evaluating whether the collected LOCAL summaries are sufficient to answer the user's question.

Rules:
- Use ONLY the provided summaries.
- Be conservative: say is_sufficient=true ONLY when the question can be fully answered from summaries.
- If not sufficient, describe what is missing and propose 1-4 follow-up queries (keywords/short phrases) that would help find missing info in remaining files.

Output JSON:
{{
  "is_sufficient": true/false,
  "knowledge_gap": "...",
  "follow_up_queries": ["..."]
}}

User question:
{question}

Summaries so far:
{summaries}
"""


answer_instructions = """Generate the final answer using ONLY the provided LOCAL summaries.

Rules (STRICT):
- Do NOT use outside knowledge.
- Do NOT invent external URLs or sources.
- If insufficient info, state clearly what is missing.
- Include citations ONLY if they appear in the summaries ([S#](local://...)).

Current date: {current_date}

User question:
{question}

Summaries:
{summaries}
"""
