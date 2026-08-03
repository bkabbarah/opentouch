# Wake-up checklist — morning of 2026-08-03

Everything below is copy-paste. Total hands-on time if all goes well: **~1 minute.**

---

## 0. Do this first, it is not optional

**Change your MIT password.** You pasted `bashark@mit.edu` and its password into
the chat last night. I did not use it — entering an account password is
something I won't do — but it is now sitting in a transcript, and if that is
your Kerberos credential it also covers the cluster and your email.

<https://atlas.mit.edu> → Password Reset. Takes a minute.

Nothing else in this file depends on it.

---

## 1. Check whether DexYCB downloaded itself (0 seconds if lucky)

DexYCB needs **no login at all** — the only thing blocking it last night was a
Google Drive anonymous-download quota, which resets on its own. A poller is
armed and retries every 30 minutes for 48 hours.

```bash
ssh bashark@mib.media.mit.edu 'tail -3 /tmp/dexycb_auto.log'
```

- Says **`SUCCESS: extracted N label files`** → go to step 2.
- Says **`quota still exceeded`** → nothing to do, it keeps trying. Skip to step 3.

## 2. If DexYCB landed, get the number (one command)

```bash
ssh bashark@mib.media.mit.edu 'bash /scratch/bashar/datasets/run_dexycb.sh'
```

This converts the labels and runs the diagnostic end to end. Read the `k=8`
median off the table it prints and compare to OpenTouch's **0.961**.

---

## 3. HO-3D — the one thing that genuinely needs you (~30 seconds)

HO-3D is the surest path to a second dataset, because its annotations contain
`handJoints3D` directly — no MANO forward pass. The catch is that it ships as
one 31.9 GB zip on OneDrive, and OneDrive's SharePoint migration blocks
headless downloads. A browser has to start it.

**Do this:**

1. Open <https://1drv.ms/f/s!AsG9HA3ULXQRlFy5tCZXahAe3bEV?e=BevrKO> in Chrome.
   (No sign-in. If it prompts you to, close the prompt — the folder is public.)
2. Click **`HO3D_v3.zip`** once to select it, then **Download** in the top toolbar.
3. Chrome starts a 31.9 GB download. Open `chrome://downloads`, **right-click
   the HO3D_v3 item → Copy link address**.
4. **Cancel** the Chrome download — you do not want it on your laptop.
5. Paste the URL into this, keeping the quotes:

```bash
ssh bashark@mib.media.mit.edu 'bash /scratch/bashar/datasets/run_ho3d.sh "PASTE_URL_HERE"'
```

That script downloads the zip to `/scratch`, extracts **only** the `meta/*.pkl`
annotations, **deletes the 31.9 GB zip**, converts to `(N,T,21,3)`, and runs the
diagnostic. Takes ~30–60 min unattended. Peak disk ~34 GB on `/scratch`, which
has 782 GB free — it never touches your home quota.

> If the URL has expired by the time you paste it (they last a few hours),
> just redo steps 1–4.

---

## 4. What I could not do, and why

| dataset | status |
|---|---|
| **DexYCB** | Fully automated, poller armed. No login needed. |
| **HO-3D** | Needs step 3. SharePoint blocks headless; no train-annotations-only archive exists. |
| **ARCTIC** | **Deprioritised.** Its cheap 215 MB drop is MANO *parameters*, not joints — needs a MANO forward pass, same problem the handoff flags for GRAB. The 18 GB "processed splits" have joints if you want it later. |

**The handoff's dataset ranking was wrong** and I'd revise it: HO-3D was listed
as "fastest path to a number" on the basis of an annotations-only download that
does not exist. Corrected order is **DexYCB → HO-3D → ARCTIC**.

---

## 4b. When `aphz` and `probesweep` finish, preserve their JSONs

Both sweeps write to the **repo root**, where `results_*.json` is gitignored on
purpose. Nothing is tracked until you copy it in — and cluster access ends
~2026-08-07, so do this before then.

```bash
ssh bashark@mib.media.mit.edu 'cd ~/scratch/bashar/opentouch-gru && grep -l . results_aperture_k[24]_ss*.json results_probe_rigid_k*_SCENE_CLEANENC.json results_probe_rigid_k*_seed*.json results_probe_rigid_k*_RANDENC.json 2>/dev/null'
```

then, for whatever that lists:

```bash
ssh bashark@mib.media.mit.edu 'cd ~/scratch/bashar/opentouch-gru && cp results_aperture_k[24]_ss*.json results_probe_*.json results/ 2>/dev/null; git add results/ && git commit -m "results: aperture horizons k=2,4 and the probe paper sweep" && git push'
```

Do **not** trust any summary of these without re-deriving the contrasts from
the JSONs — that is what turned up corrections 9–11 last night.

---

## 5. What ran while you slept

- **`rotshare` finished and PASSED.** The agnostic diagnostic gives 0.961 at
  k=8 against the 0.957 reference — it agrees, so cross-dataset runs are
  trustworthy. Details and a caveat in `HANDOFF.md` §2.25.
- **The by-scene breakdown is a better result than the single number.** All
  **26 scenes above 90%** rigid share, range 90.5%–99.3%. That upgrades "we
  measured this once" to "this holds across every activity we have."
- **§2.22–§2.24 re-derived from the JSONs.** They hold. Three small doc
  corrections applied — see §3 of `HANDOFF.md`.
- **`aphz` still running**, ~8/24 runs done as of 23:43, on k=4 now.
  `probesweep` correctly queued behind it.
- **New: `scripts/load_public_hands.py`** + 15 tests. Suite is **258 passing**
  (was 243).

```bash
ssh bashark@mib.media.mit.edu 'tail -3 /tmp/aphz_master.log; tail -2 /tmp/probesweep_master.log'
```

### One housekeeping note

To pull the new code onto the cluster I ran `git stash -u`, and the matching
`git stash drop` was blocked by a permission guard, so **`stash@{0}` is still
sitting in the cluster repo**. It is redundant — every file in it is either
restored in the working tree or now tracked in git, and `git stash show` on it
reports zero tracked changes. Clear it whenever you like:

```bash
ssh bashark@mib.media.mit.edu 'cd ~/scratch/bashar/opentouch-gru && git stash drop stash@{0}'
```

`stash@{1}` is older than this session and is not mine — leave it alone unless
you know what it is.
