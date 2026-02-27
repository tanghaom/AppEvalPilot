# Findings: api_log/136

## Directory scan
- `appeval/api_log/136` currently includes at least 92 `info.txt` files (many app-case runs).
- The dominant artifacts are `info.txt`, `test_case.json`, and per-step images (for many runs).

## High-severity recurring failures
- **App not available / wrong app state**:
  - `App Not Ready` appears heavily in multiple runs (e.g., 90s Retro Business Card and VideoClipper families).
  - Visual evidence: `draw_*.jpg` often shows only "Your app is not ready yet" or blank Chrome.
- **Route resolution failures**:
  - Multiple runs show `Error` page with `no Route matched with those values`.
  - Example image: `ArtisticColoringBook0/202602261126/draw_2.jpg`.
- **Window/session instability**:
  - `No tab matches expected domain` appears repeatedly (high-frequency across many logs).
  - `No valid window found to extract elements from` appears in affected runs.
  - `Display connection closed by server: [Errno 32] Broken pipe` repeats after failed actions.
- **Screenshot pipeline failures**:
  - `Screenshot failed: cannot identify image file '/tmp/tmp....png'`.
  - Followed by `All 3 attempts failed for task test_case` and missing `.../screenshot/screenshot.jpg`.

## Data consistency issue in result files
- Many `test_case.json` records store failure evidence as:
  - `Error: [Errno 2] No such file or directory: '.../screenshot/screenshot.jpg'`.
- Evidence paths often point to a different sibling run directory index (e.g., `...Generator3/...`) rather than current case context, suggesting retry/run-index path binding issues.

## Representative files
- `appeval/api_log/136/202602261125/Language Spelling Bee/Language Spelling Bee2/202602261131/info.txt`
- `appeval/api_log/136/202602261125/MenuExpress/MenuExpress0/202602261133/info.txt`
- `appeval/api_log/136/202602261125/90s Retro Business Card/90s Retro Business Card0/202602261126/draw_1.jpg`
- `appeval/api_log/136/202602261125/ArtisticColoringBook/ArtisticColoringBook0/202602261126/draw_2.jpg`
