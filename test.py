import os, zipfile
root = "/inspire/ssd/project/robotsimulation/guchun-240107140023/Wenyao/calvin_dense_k10_action_5/task_ABC_D/"
bad = []
for dp,_,fs in os.walk(root):
    for f in fs:
        if f.endswith(".npz"):
            p = os.path.join(dp,f)
            try:
                with zipfile.ZipFile(p,'r') as zf:
                    if zf.testzip() is not None:
                        print("[BAD-ZIP]", p)
                        bad.append(p)
            except Exception as e:
                print("[BROKEN]", p, e); bad.append(p)
print("TOTAL BAD:", len(bad))