COMMAND_PROMPT = (
    "Return valid JSON only. No markdown. "
    "Expand and simplify the polynomial expression. "
    "Give concise step-by-step work."
)

CONTENT_PROMPT = """Task information:
example_id: {example_id}
question: {problem_sentence}
"""

TEMPLATE_PROMPT = """Return one JSON object with exactly these fields:
{
  "question": "...",
  "steps": ["...", "..."],
  "final_answer": "..."
}

Rules:
- Use the exact question above in the question field.
- This is a simplification task, not an equation-solving task.
- Expand products and combine like terms.
- Keep coefficients exact; do not use decimal approximations.
- Put all math in steps inside $...$.
- Use 2-5 concise steps, depending on the problem.
- The final step must include the simplified expression inside $\\boxed{...}$.
- final_answer must contain only the simplified expression, without $...$ or \\boxed{}.
- In final_answer, use * for multiplication and ^ for powers.
"""