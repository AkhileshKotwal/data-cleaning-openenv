# Ops Workbench Task Validation Fix - TODO

## Steps from Approved Plan
- [ ] 1. Edit inference.py: Clamp mean_score to (0.01, 0.98)
- [ ] 2. Edit environment.py: Extra clamp on safe_score
- [ ] 3. Edit tasks.py: Add clamping comment
- [ ] 4. Run `python inference.py` to test
- [ ] 5. Verify: `grep -E \"0\.0|1\.0\" results.json`
- [ ] 6. attempt_completion

Progress: Plan approved, starting edits.
