import os, subprocess, mlx.core as mx
a = mx.ones((1_000_000_000,), dtype=mx.float32); mx.eval(a)   # exactly 4.0e9 bytes of GPU memory
pid = os.getpid()
print("allocated_bytes", mx.get_active_memory(), " = %.4f GiB / %.4f GB(dec)" % (mx.get_active_memory()/2**30, mx.get_active_memory()/1e9))
for fmt in ("formatted","bytes"):
    out = subprocess.run(["footprint","-f",fmt,"-p",str(pid)],capture_output=True,text=True).stdout
    print("--- footprint -f", fmt, "---")
    print("\n".join(l for l in out.splitlines() if "Footprint" in l or "phys_footprint" in l or "IOAccelerator (graphics)" in l))
