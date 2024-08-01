from jason_HN_time import hopfield

myhopfield = hopfield(9)
print(myhopfield.time_steps)
myhopfield.propagate(0)

print(myhopfield.dou_factor)
print(myhopfield.propagated)