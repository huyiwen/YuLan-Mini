# Estimate Model FLOPs Utilization of YuLan-Mini stable training stage

D = 25 * 40 * 10 ** 9

N1 = 56
t1 = 10 * 28 * 60 * 60  # 10 stages, 23 hours/stage

N2 = 48  # shrink the cluster size
t2 = 15 * 32 * 60 * 60  # 15 stages, 32 hours/stage

T = D / (N1 * t1 + N2 * t2)
print("T =", T)

C = 312 * 10 ** 12  # A800 GPU chips
B = 1008  # = 56 * 18 = 46 * 21
s = 4096  # seq length
l = 56  # num hidden layers
h = 1920  # hidden size
f = 4800  # intermediate size
V = 99000  # vocab size

E = 8 * B * s * l * h ** 2 + 6 * B * s * l * h * f + 4 * B * s ** 2 * l * h
F = 3 * E + 4 * B * s ** 2 * l * h + 6 * B * s * h * V

print("F =", F)

MFU = F * T / B / s / C
print("MFU =", MFU)
