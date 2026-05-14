# Test Data Policy

This directory keeps small smoke fixtures and deterministic data generators.

## Tracked fixtures

- `test_1d.npy`
- `test_2d.npy`
- `test_data.npz`

These files are intentionally tiny and are used by `verify_functions.py`.

## Generated UX data

Run:

```powershell
python test_data/create_uxcase_test_data.py
python test_data/verify_uxcase_data.py
```

This creates semantic files with the `uxcase_` prefix plus
`uxcase_manifest.json`. Generated `.npy/.npz` files are ignored by git.

## Local large data

Very large files such as F3 seismic cubes should live outside git, for example:

```text
local_data/
```

If public download is needed, attach those files to GitHub Releases or use Git
LFS intentionally.
