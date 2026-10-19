"""Streamlit viewer for top-k audio<->video retrievals produced by dump_retrievals.py.

    streamlit run app_retrievals.py --server.port 8501 -- \
        --results retrievals_e65.pkl --split test \
        --media_root /path/to/folder/containing/how2_*-segments

Every clip is shown as the .mp4 with its .wav muxed in (via ffmpeg, cached on disk).
The next page is muxed in the background while you look at the current one.
"""
import argparse
import hashlib
import json
import os
import pickle
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import streamlit as st

# Results pickled under numpy>=2 reference `numpy._core.*`; numpy 1.x only has `numpy.core.*`.
if not hasattr(np, "_core") and "numpy._core" not in sys.modules:
    import numpy.core
    import numpy.core.multiarray
    import numpy.core.numeric
    sys.modules["numpy._core"] = numpy.core
    sys.modules["numpy._core.multiarray"] = numpy.core.multiarray
    sys.modules["numpy._core.numeric"] = numpy.core.numeric

st.set_page_config(page_title="AV Retrieval Viewer", layout="wide")


# --------------------------------------------------------------------------- args
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--results", default="retrievals.pkl", help="Output of dump_retrievals.py")
    p.add_argument("--split", default="test", choices=["val", "test"], help="Which split to show")
    p.add_argument("--media_root", default=None,
                   help="Folder that contains the how2_*-video-segments / *-audio-segments folders. "
                        "Defaults to the folder containing the manifest .json.")
    p.add_argument("--per_page", type=int, default=3, help="Samples per page (3 or 4 recommended)")
    p.add_argument("--cache_dir", default=str(Path.home() / ".cache" / "av_retrieval_viz"),
                   help="Where muxed video+audio mp4s are cached (use a local disk).")
    p.add_argument("--reencode", action="store_true",
                   help="Re-encode video to H.264 (use if videos show black / don't play in the browser).")
    p.add_argument("--workers", type=int, default=8, help="Parallel ffmpeg jobs.")
    args, _ = p.parse_known_args()
    return args


ARGS = parse_args()
FFMPEG = shutil.which("ffmpeg")
CACHE = Path(ARGS.cache_dir) / ("mux_h264" if ARGS.reencode else "mux_copy")
CACHE.mkdir(parents=True, exist_ok=True)

DIRECTIONS = {
    "a2v": "Audio → Video  (query = speech, retrieve videos)",
    "v2a": "Video → Audio  (query = video, retrieve speech)",
}


# --------------------------------------------------------------------------- compat (old Streamlit)
def divider():
    if hasattr(st, "divider"):
        st.divider()
    else:
        st.markdown("---")


def bordered():
    try:
        return st.container(border=True)
    except TypeError:
        return st.container()


def caption(text, help=None):
    try:
        st.caption(text, help=help)
    except TypeError:
        st.caption(text)


# --------------------------------------------------------------------------- loading
# cache_resource (not cache_data) everywhere: no hashing / copying of big objects on each rerun.
@st.cache_resource
def load_split(results_path, split, media_root):
    with open(results_path, "rb") as f:
        res = pickle.load(f)
    if split not in res["splits"]:
        return None, f"Split `{split}` not in results file (has: {list(res['splits'])})."
    S = res["splits"][split]
    manifest_path = S.get("manifest_path")
    if manifest_path is None:
        return None, f"No manifest path stored for split `{split}`."

    txt = Path(manifest_path).read_text()
    rows = json.loads(txt) if txt.lstrip().startswith("[") else \
        [json.loads(l) for l in txt.splitlines() if l.strip()]

    n = S["n"]
    entries, how, warn = None, None, None
    by_stem = {}
    for e in rows:
        for f in ("video_path", "audio_path"):
            if e.get(f):
                by_stem[Path(e[f]).stem] = e
    for k, vals in (S.get("keys") or {}).items():
        if len(vals) == n:
            out = [by_stem.get(Path(str(v)).stem) for v in vals]
            if sum(o is not None for o in out) >= 0.9 * n:
                entries, how = out, f"via dataset field `{k}`"
                break
    if entries is None:
        if len(rows) == n:
            entries, how = rows, "by manifest order"
        else:
            entries, how = (rows + [None] * n)[:n], "manifest order (UNSAFE)"
            warn = (f"Manifest has {len(rows)} rows but {n} embeddings — labels may be misaligned. "
                    f"Make the dataset return `video_path` and re-run dump_retrievals.py.")

    root = Path(media_root) if media_root else Path(manifest_path).parent
    media = []
    for e in entries:
        if e is None:
            media.append((None, None))
            continue
        v, a = e.get("video_path"), e.get("audio_path")
        media.append(((root / v) if v else None, (root / a) if a else None))

    return dict(S=S, entries=entries, media=media, manifest_path=manifest_path, how=how, warn=warn,
                pretrain_path=res.get("pretrain_path")), None


@st.cache_resource
def executor():
    return ThreadPoolExecutor(max_workers=ARGS.workers)


# --------------------------------------------------------------------------- muxing
def mux_target(video, audio):
    return CACHE / (hashlib.md5(f"{video}|{audio}".encode()).hexdigest() + ".mp4")


def mux(video, audio):
    """Return path to an mp4 with `audio` as its soundtrack (cached). None if not possible."""
    if video is None or FFMPEG is None or audio is None:
        return None
    out = mux_target(video, audio)
    if out.exists() and out.stat().st_size > 0:
        return out
    if not video.exists() or not audio.exists():
        return None
    vcodec = (["-c:v", "libx264", "-preset", "veryfast", "-crf", "26", "-pix_fmt", "yuv420p"]
              if ARGS.reencode else ["-c:v", "copy"])
    tmp = out.with_name(f"{out.stem}.{os.getpid()}.{os.urandom(4).hex()}.tmp.mp4")
    cmd = [FFMPEG, "-y", "-loglevel", "error", "-i", str(video), "-i", str(audio),
           "-map", "0:v:0", "-map", "1:a:0", *vcodec, "-c:a", "aac", "-b:a", "96k",
           "-shortest", "-movflags", "+faststart", str(tmp)]
    try:
        subprocess.run(cmd, check=True, capture_output=True, timeout=120)
        tmp.replace(out)
        return out
    except Exception:
        try:
            tmp.unlink()
        except OSError:
            pass
        return None


def clips_for(qs, top_idx):
    return sorted({int(i) for q in qs for i in [q, *top_idx[q]]})


def mux_many(idxs, media, wait):
    futs = [executor().submit(mux, *media[i]) for i in idxs
            if media[i][0] is not None and media[i][1] is not None
            and not mux_target(*media[i]).exists()]
    if wait:
        for f in futs:
            f.result()


# --------------------------------------------------------------------------- rendering
def show_clip(i, media):
    v, a = media[i]
    if v is not None and a is not None:
        m = mux_target(v, a)
        if m.exists():
            st.video(str(m))
            return
    if v is not None and v.exists():
        st.video(str(v))
    if a is not None and a.exists():
        st.audio(str(a))
    if (v is None or not v.exists()) and (a is None or not a.exists()):
        st.error(f"Media not found: `{v}`")


def short(text, n=110):
    text = (text or "").strip()
    return text if len(text) <= n else text[: n - 1] + "…"


def rank_badge(r):
    return "🟢" if r == 1 else "🟡" if r <= 5 else "🟠" if r <= 10 else "🔴"


# --------------------------------------------------------------------------- app
if not Path(ARGS.results).exists():
    st.error(f"Results file `{ARGS.results}` not found. Run dump_retrievals.py first.")
    st.stop()

data, err = load_split(ARGS.results, ARGS.split, ARGS.media_root)
if err:
    st.error(err)
    st.stop()
S, entries, media = data["S"], data["entries"], data["media"]

with st.sidebar:
    st.header(f"{'Validation' if ARGS.split == 'val' else 'Test'} split")
    direction = st.radio("Direction", list(DIRECTIONS), format_func=DIRECTIONS.get)
    order = st.selectbox("Order samples by",
                         ["Dataset order", "Random", "GT rank (best first)", "GT rank (worst first)"])
    show_only = st.selectbox("Show only", ["All", "Hits (GT in top-10)", "Misses (GT not in top-10)",
                                           "Top-1 correct"])
    seed = int(st.number_input("Random seed", value=0, step=1)) if order == "Random" else 0
    per_page = int(st.number_input("Samples per page", min_value=1, max_value=10,
                                   value=ARGS.per_page, step=1))

D = S[direction]
top_idx, top_sc, gt = D["top_idx"], D["top_scores"], D["gt_rank"]
K = top_idx.shape[1]

q = np.arange(S["n"])
if show_only.startswith("Hits"):
    q = q[gt[q] <= 10]
elif show_only.startswith("Misses"):
    q = q[gt[q] > 10]
elif show_only.startswith("Top-1"):
    q = q[gt[q] == 1]
if order == "Random":
    q = np.random.default_rng(seed).permutation(q)
elif order == "GT rank (best first)":
    q = q[np.argsort(gt[q], kind="stable")]
elif order == "GT rank (worst first)":
    q = q[np.argsort(-gt[q], kind="stable")]
n_pages = max(1, int(np.ceil(len(q) / per_page)))

# ---- page state: reset when the filters change, clamp to range, then build the widgets
sig = (direction, order, show_only, seed, per_page)
if st.session_state.get("sig") != sig:
    st.session_state["sig"] = sig
    st.session_state["page"] = 1
st.session_state["page"] = int(min(max(1, st.session_state.get("page", 1)), n_pages))


def go(delta):
    st.session_state["page"] = int(min(max(1, st.session_state["page"] + delta), n_pages))


with st.sidebar:
    st.number_input(f"Page (1–{n_pages})", min_value=1, max_value=n_pages, step=1, key="page")
    jump = st.text_input("Jump to sample index", "")
    st.caption(f"{len(q)} samples · {n_pages} pages")
    divider()
    st.caption(f"Checkpoint: `{Path(str(data['pretrain_path'])).name}`")
    st.caption(f"Manifest: `{data['manifest_path']}`")
    st.caption(f"Alignment: {data['how']}")
    if FFMPEG is None:
        st.warning("ffmpeg not found — showing video and audio as separate players.")

page = st.session_state["page"]

# ---- header
st.title(f"{'Validation' if ARGS.split == 'val' else 'Test'} · {DIRECTIONS[direction]}")
if data["warn"]:
    st.warning(data["warn"])
r = gt.astype(np.float64)
c = st.columns(5)
c[0].metric("R@1", f"{(r <= 1).mean() * 100:.2f}%")
c[1].metric("R@5", f"{(r <= 5).mean() * 100:.2f}%")
c[2].metric("R@10", f"{(r <= 10).mean() * 100:.2f}%")
c[3].metric("MRR", f"{(1 / r).mean():.4f}")
c[4].metric("Median rank", f"{np.median(r):.0f}")
st.caption("🟢 GT at rank 1 · 🟡 ≤5 · 🟠 ≤10 · 🔴 >10.   ✅ = correct pair · 🔗 = same source YouTube video as the query.")


def pager(where):
    a, b, c_ = st.columns([1, 3, 1])
    a.button("◀ Prev", key=f"prev_{where}", on_click=go, args=(-1,), disabled=page <= 1)
    b.markdown(f"<div style='text-align:center;padding-top:6px'>Page <b>{page}</b> / {n_pages}</div>",
               unsafe_allow_html=True)
    c_.button("Next ▶", key=f"next_{where}", on_click=go, args=(1,), disabled=page >= n_pages)


pager("top")

if jump.strip().isdigit() and int(jump) < S["n"]:
    page_q = [int(jump)]
else:
    page_q = [int(x) for x in q[(page - 1) * per_page: page * per_page]]

# Mux this page (blocking), then the next page in the background so "Next" is instant.
with st.spinner("Preparing clips…"):
    mux_many(clips_for(page_q, top_idx), media, wait=True)
next_q = [int(x) for x in q[page * per_page: (page + 1) * per_page]]
mux_many(clips_for(next_q, top_idx), media, wait=False)

for qi in page_q:
    e = entries[qi] or {}
    rk = int(gt[qi])
    with bordered():
        st.markdown(f"#### {rank_badge(rk)} Sample `{qi}` — correct match at rank **{rk}** / {S['n']}")
        left, right = st.columns([1, 2])
        with left:
            show_clip(qi, media)
        with right:
            st.markdown(f"**Transcript:** {e.get('text', '—')}")
            st.markdown(f"**Video caption:** {e.get('video_caption', '—')}")
            extra = f" · denseav {float(e['denseav_score']):.3f}" if "denseav_score" in e else ""
            st.caption(f"video `{e.get('original_video_id', '?')}` · {e.get('start', '?')}–{e.get('end', '?')}s · "
                       f"speaker `{e.get('speaker_id', '?')}` · lm_score {e.get('lm_score', '?')}{extra}")

        st.markdown(f"**Top-{K} retrievals**")
        for row_start in range(0, K, 5):
            cols = st.columns(5)
            for j, col in enumerate(cols):
                k = row_start + j
                if k >= K:
                    break
                ci = int(top_idx[qi, k])
                ce = entries[ci] or {}
                is_gt = ci == qi
                same_src = (not is_gt and ce.get("original_video_id")
                            and ce.get("original_video_id") == e.get("original_video_id"))
                tag = " ✅" if is_gt else (" 🔗" if same_src else "")
                with col:
                    st.markdown(f"**#{k + 1}**{tag} · `{top_sc[qi, k]:.3f}`")
                    show_clip(ci, media)
                    caption(short(ce.get("text")), help=(
                        f"Transcript: {ce.get('text', '—')}\n\n"
                        f"Caption: {ce.get('video_caption', '—')}\n\n"
                        f"Sample {ci} · video {ce.get('original_video_id', '?')} · "
                        f"speaker {ce.get('speaker_id', '?')}"))
    divider()

pager("bottom")