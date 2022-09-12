from melkor_logic.NLM import *
import random

config = [[1,3,1],[10,10,5],[1,3,1]]
net = NeuroLogicMachine(config)

N = 4;B = 6

def make_p(inputs,batch,loc,TF):
    inputs[batch][loc] = TF

def make_q(inputs,batch,n,loc,TF):
    inputs[batch][n][loc] = TF

def make_r(inputs,batch,n1,n2,loc,TF):
    inputs[batch][n1][n2][loc] = TF

def make_testcase(b,n):
    P = torch.zeros([B,1]);Q = torch.zeros([B,N,3])
    R = torch.zeros([B,N,N,1])
    for b in range(B):
        p_flag = int(random.random()+0.5)
        n1 = random.randint(0,N-1);n1_flag = int(random.random()+0.5)
        n2 = random.randint(0,N-1);n2_flag = int(random.random()+0.5)
        make_p(P,b,0,p_flag)
        make_q()
    outputs = [P,Q,R]
    return outputs

tc = make_testcase(B,N)

"""
P = torch.zeros([b,1])
Q = torch.zeros([b,N,3])
R = torch.zeros([b,N,N,1])
# if the condition P is wrong, then it is wrong
make_p(P,0,0,0);make_p(P,1,0,0)
make_p(P,2,0,0);make_p(P,3,0,0)
make_p(P,4,0,0);make_p(P,5,0,0)
make_p(P,6,0,1);make_p(P,7,0,1)
make_p(P,8,0,1);make_p(P,9,0,1)
make_p(P,10,0,1);make_p(P,11,0,1)
# if any 0 or 1 term of Q is correct, then it is correct
make_q(Q,0)

tc = [
    P,Q,R
] # in the test case 1, no relational logic are included
"""

ts = [
    torch.zeros([b,1]),torch.zeros(torch.zeros([b,N,3])),torch.zeros([b,N,N,1])
]

output = net(tc)