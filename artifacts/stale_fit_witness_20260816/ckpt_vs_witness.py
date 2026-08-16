import os, glob, re, json, sys
R = "/scratch/midway3/rickybao/dreamerv3_runs"
out = []
for d in sorted(glob.glob(os.path.join(R, "ax1wm_finger_*s12*")) +
                glob.glob(os.path.join(R, "ax1wm_finger_*s25*"))):
    kv = {}
    try:
        for line in open(os.path.join(d, "OFFLINE_FIT_PROGRESS")):
            if "=" in line:
                k, v = line.strip().split("=", 1); kv[k] = v
    except OSError:
        continue
    try:
        upd = int(kv.get("update", 0)); tot = int(kv.get("total_updates", 0))
    except ValueError:
        continue
    ck = []
    for c in glob.glob(os.path.join(d, "ckpt", "*")):
        m = re.search(r"^(\d{8}T\d{6})F\d+-(\d+)$", os.path.basename(c))
        if m and os.path.exists(os.path.join(c, "done")):
            ck.append((m.group(1), int(m.group(2))))
    ck.sort()
    top = [c for c in ck if c[1] >= tot] if tot else []
    out.append({
        "run": os.path.basename(d),
        "witness_update": upd, "total": tot,
        "witness_at": kv.get("updated_at", ""),
        "n_done_ckpt": len(ck),
        "first_done": ck[0] if ck else None,
        "last_done": ck[-1] if ck else None,
        "done_at_total": top[0][0] if top else None,
    })
print(json.dumps(out, indent=1))
