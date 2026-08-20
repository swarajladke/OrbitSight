# Agent Operating Rules — OrbitSight

These rules override any instinct to be helpful, thorough, or complete.
Violating them is worse than doing nothing.

## Rule 1 — Never claim you did something you did not do

- You may report a change as done ONLY if it appears in a file you edited this session.
- You MUST NOT report any result that requires running a command unless you actually ran it
  and are pasting its real output.
- If you cannot run a command, write exactly: `NOT RUN — requires user execution`.
- Never write "Completed", "Verified", "Confirmed", or "Measured" about anything you did not
  observe directly in this session.

## Rule 2 — Never invent numbers

- Do not write any numeric value into any file or message unless it came from real command
  output in this session, or from a file you read in this session.
- If a value is unknown, write the literal token `UNMEASURED`. Nothing else.
- FORBIDDEN: placeholder tokens that look like variables (`ub_rec`, `hist[0.0]`, `prec`).
  FORBIDDEN: plausible-looking invented statistics.
- If you write a table, every cell is either a real measured value or `UNMEASURED`.

## Rule 3 — Conclusions come after evidence, never before

- Do not write a "Findings", "Root Cause", "Answer", or "Conclusion" section unless the
  supporting data in the same document is fully populated with measured values.
- If the data is `UNMEASURED`, the conclusion section must read exactly:
  `PENDING — no data collected yet.`

## Rule 4 — Quote before you edit

Before editing any existing function, paste the current code of that function verbatim into
your response. If you cannot paste it, you have not read it, and you must read it first.
Never edit code you have not quoted.

## Rule 5 — Scope discipline

- Change ONLY files listed in the task's "Files you may modify" section.
- If a task seems to require touching another file, STOP and report:
  `BLOCKED — task requires modifying <file>, which is not in scope.`
- Do not add features, helpers, CLI flags, refactors, or "improvements" that were not
  explicitly requested. Do not rename anything. Do not reformat unrelated lines.
- Do not create new files unless the task names the file to create.

## Rule 6 — Do not fix forward silently

If a command fails, paste the full traceback. Do not attempt more than one fix without
showing the user the failure first.

## Rule 7 — Mandatory self-check

End EVERY response with this block, filled honestly:

```
SELF-CHECK
Files modified: <exact list, or NONE>
Commands actually executed this session: <exact list, or NONE>
Numbers in this response that came from real output: <list, or NONE>
Numbers I could not verify: <list, or NONE>
Claims marked NOT RUN: <list, or NONE>
Out-of-scope changes made: <list, or NONE — must be NONE>
```

If any line is inaccurate, you have failed the task regardless of code quality.

## Rule 8 — Project facts you must not contradict

- Event `.npy` columns are: 0=x, 1=y, 2=polarity, 3=timestamp_us, 4=label,
  5=relative_timestamp_us. The timestamp is column 3, NOT column 2.
- Window size is 40,000 us. IoU threshold for a match is 0.5.
- `iter_windows` uses `events[:, 3]`. Any code using `events[:, 2]` as time is a bug.
- There is exactly ONE IoU implementation: `src.metrics.iou`. Never write another.
- There is exactly ONE pipeline entry point: `src.pipeline.run_sequence`, which returns
  a 2-tuple `(predictions, num_windows)`.
- Tuning and reporting use the 17 TRAINING sequences only. The 4 test sequences are never
  used to select a parameter.
  ## Rule 9 — This protocol applies to every task automatically

Any message beginning with `TASK:` is governed by Rules 1–8 without restating them.
You must apply the following to EVERY such message, even when it is not mentioned:

1. **Scope is closed.** Modify only files listed after `SCOPE:`. If the message has no
   `SCOPE:` line, ask which files are in scope before editing anything. Never infer scope.
2. **Quote before editing.** Paste the current code of each function you are about to change.
3. **No additions.** No new files, flags, helpers, refactors, renames, or reformatting
   beyond the literal instruction.
4. **Run the `CHECK:` command** if one is given, and paste its raw output.
5. **End with a verdict**, using exactly one of these strings and no other wording:
   `PASS — command ran, output pasted above`
   `FAIL — command ran, traceback pasted above`
   `NOT RUN — could not execute`
6. **End with the SELF-CHECK block** from Rule 7.

If a task cannot be completed within `SCOPE:`, reply only:
`BLOCKED — requires modifying <file>, not in scope.`