from melkorvision import *
from melkor_logic import *
from melkor_logic.TOQNet import *

net = TOQNet([(3,12,3),(8,24,5),(20,4,5)],[24,28,32])

N = 10;b = 3;T = 10
P = torch.randn([b,T,3])
Q = torch.randn([b,T,N,12])
R = torch.randn([b,T,N,N,3])
inputs = (P,Q,R)
Pc = net(*inputs)
print(Pc.shape)

from melkor_logic.NLM.model import NeuroLogicMachine

config = ([12,14,10],[18,4,5])
nlm = NeuroLogicMachine(config)

conclusions = nlm(torch.randn(b,12),torch.randn(b,N,14),torch.randn(b,N,N,10))
print(conclusions[0])