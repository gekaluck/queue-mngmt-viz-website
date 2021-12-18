import numpy


def normalGenerat(a, size):                               # GENERATORS
        a = numpy.random.normal(a[0], a[1], size)
        return numpy.absolute(a)

def logNormalGenerate(a, size):                           # GENERATORS
    a = numpy.random.normal(a[0], a[1], size)
    if not a[0] == 0:
        return numpy.exp(a)

Tp = 2
Tp_sd = 1
ia_t = 3
ia_t_sd = 1

Rate = [[Tp, Tp_sd], [ia_t, ia_t_sd]]
#ARate = [Tp, ia_t]

#ADistribution = [numpy.random.exponential]

# What size?????????????????

# 2*Time/a +

# print("logNormal")
# phiA = numpy.sqrt(ia_t ** 2 + ia_t_sd ** 2)
# phiP = numpy.sqrt(Tp ** 2 + Tp_sd ** 2)
# ARate = [[numpy.log((Tp ** 2) / phiP), numpy.sqrt(numpy.log((phiP ** 2) / (Tp ** 2)))],
#          ### ?????????????? [should be [Tp, Tp_sd], [ia_t, ia_t_sd]]
#          [numpy.log((ia_t ** 2) / phiA), numpy.sqrt(numpy.log((phiA ** 2) / (ia_t ** 2)))]]  # ??????????????
# print("[Tp, Tp_sd], [ia_t, ia_t_sd]", ARate)
# ADistribution = [logNormalGenerate]
# print(ADistribution[0](ARate[1], size=10))
#
# print("Normal")
# ARate = Rate
# cva = ia_t_sd
# ADistribution = [normalGenerat]
# print("[Tp, Tp_sd], [ia_t, ia_t_sd]", ARate)
# print(ADistribution[0](ARate[1], size=10))

print("Exponential")
ARate = [Tp, ia_t]
ADistribution = numpy.random.exponential(ARate[1], size=15)
print(ADistribution)
# print("[Tp, Tp_sd], [ia_t, ia_t_sd]", ARate)

print("Normal")
ARate = Rate
ADistribution = normalGenerat(ARate[1], size=20)
print(ADistribution)

print("logNormal")


print("Loop")
index = 0
for i in range(30): # Just for demonstration of environment time flow
    if i % 3 == 0:
        print(ADistribution[index])
        index += 1