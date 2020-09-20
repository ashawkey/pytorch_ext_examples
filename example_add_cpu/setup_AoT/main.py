from add import add, Add
import torch

'''
static setup, run first:
python setup.py install
'''


one = torch.ones(1)

adder = Add()
print(adder(one, one))

print(add(one, one))
