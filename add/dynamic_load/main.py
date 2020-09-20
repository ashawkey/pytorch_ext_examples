from add import add, Add
import torch


'''
dynamic load saves compiled results in __pycache__, so don't delete it.
'''

one = torch.ones(1)

adder = Add()
print(adder(one, one))

print(add(one, one))
