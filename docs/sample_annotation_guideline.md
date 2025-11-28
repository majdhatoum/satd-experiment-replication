# SATD Annotation Guideline (v1.0)

This short guide explains how to label Self-Admitted Technical Debt (SATD) in short software-engineering texts such as code comments, commit messages, issues, and pull-request descriptions. It defines the task, the labels, and a few “decision rules” so different annotators work consistently.

> Grounding: Our definitions follow prior SATD work (e.g., Potdar & Shihab 2014; Maldonado & Shihab 2015; Chen et al. 2021; Li et al. 2023) with minor operational tweaks for this study.

---

## 1) What you will see

Each row is one short text snippet (a code comment line or block, a commit/issue/PR message). For each item, please provide:

- **SATD (yes/no/not-sure)** — does the text admit or clearly imply a technical compromise or debt?
- **SATD Type** (only if SATD = yes) — choose *one*:
  - `design-debt`
  - `requirement-debt`
  - `documentation-debt`
  - `test-debt`
  
Please label based only on the text shown; do not search the web or additional context.

---

## 2) “What counts as SATD?”

An item is **SATD** if it *explicitly acknowledges* a technical compromise, limitation, workaround, shortcut, or the intention/need to fix/improve something later.

Common cues (not exhaustive):

- “TODO”, “FIXME”, “hack”, “workaround”, “temporary”, “tech debt”, “refactor later”
- “should”, “need to”, “won’t fix now”, “for now we…”
- Naming a **limitation** or **risk** that the team accepts temporarily

**Non-SATD** examples:

- Neutral statements (“Add docs for X”, “Reformat code”)
- Pure bug reports with no self-admission of a compromise (“Crash when clicking X”)
- Opinion/plan with no stated compromise (“Consider feature Y”)

If it’s unclear or reads like a generic plan/bug without self-admission, choose **no** (or **not-sure** if truly ambiguous).

---

## 3) SATD Types (choose exactly one when SATD = yes)

### 3.1 Design-Debt
Code/architecture design is sub-optimal; needs refactoring, redesign, or is using a quick hack.

- Cues: “refactor”, “rename”, “restructure”, “architecture”, “hacky”, “tight coupling”
- Example (generic): “Quick workaround to meet the deadline — should refactor to a proper service.”

### 3.2 Requirement-Debt
Functionality/behavior is knowingly incomplete, postponed, or implemented as a stop-gap vs. stated/implicit requirements.

- Cues: “not fully implemented”, “missing feature”, “temporary behavior”, “TODO: support X”
- Example: “For now we handle only CSV; TODO: add JSON and XML.”

### 3.3 Documentation-Debt
Documentation (code comments, API docs, user/developer guides) is missing, outdated, or intentionally postponed.

- Cues: “docs are missing/outdated”, “update README later”, “need examples/tutorial”
- Example: “This API changed; docs will be updated after release.”

### 3.4 Test-Debt
Testing is missing/insufficient/flaky; tests intentionally postponed or simplified.

- Cues: “no tests”, “skip flaky test”, “TODO: add integration tests”
- Example: “Skipping edge cases here; tests to be added after refactor.”

> **Boundary rule:** If text clearly fits two types, pick the **dominant intent**. If still tied, prefer **design-debt** when the statement is about code structure/quality; prefer **requirement-debt** when it’s about incomplete behavior.

---

## 4) Decision quick-flow

1. **Does the text *admit a compromise* explicitly?**  
   - Yes → SATD = **yes** → go to type selection  
   - No → SATD = **no**  
   - Unsure/ambiguous wording → **not-sure**
2. **If SATD = yes, pick one type** from §3 using the boundary rule above.
---

## 5) Edge cases & tie-breakers

- “Temporary workaround” for a bug **without** mentioning design/code → usually **requirement-debt** (incomplete behavior) unless it clearly references refactoring/architecture → **design-debt**.
- “Will document later” → **documentation-debt** even if the change also hints at test gaps.
- “Add tests later” → **test-debt** even if a feature is also incomplete.
- If a snippet is meta-discussion (e.g., triage boilerplate) with no self-admission → **no**.

---

## 6) Quality expectations

- Be consistent: apply the same decision rules across items.
- Use **not-sure** sparingly when wording is truly ambiguous.
- Do **not** copy text out of the sheet or share it externally. Treat items as confidential research material.

---

## 7) Annotation format

Your sheet has the following columns (headers may be pre-filled):


```bash
item_id, source_type, project, origin_ref, text, stratum, SATD, SATD Type
```

- `source_type` ∈ {commit, code_comment, issue, pull_request}
- `SATD` ∈ {yes, no, not-sure}
- `SATD Type` only if `SATD=yes` ∈ {design-debt, requirement-debt, documentation-debt, test-debt}

---

## 8) Time & contact

- Pilot/training: ~10 minutes (examples + Q&A)
- Main session: ~45–60 minutes for 32 items
- Questions: contact the study lead (see consent sheet)

*Thank you for contributing!*
