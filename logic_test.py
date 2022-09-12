from melkor_logic.NLM import *
import random

config = [[1,3,1],[10,10,5],[1,3,1]]
net = NeuroLogicMachine(config)

N = 30;B = 100

def make_p(inputs,batch,loc,TF):
    inputs[batch][loc] = TF

def make_q(inputs,batch,n,loc,TF):
    inputs[batch][n][loc] = TF

def make_r(inputs,batch,n1,n2,loc,TF):
    inputs[batch][n1][n2][loc] = TF

def make_testcase(b,n):
    P = torch.zeros([B,1]);Q = torch.zeros([B,N,3])
    R = torch.zeros([B,N,N,1])
    outputs = []
    for b in range(B):
        p_flag = int(random.random()+0.5)
        n1 = random.randint(0,N-1);n1_flag = int(random.random()+0.5)
        n2 = random.randint(0,N-1);n2_flag = int(random.random()+0.5)
        make_p(P,b,0,p_flag)
        make_q(Q,b,n1,0,n1_flag);make_q(Q,b,n2,1,n2_flag)
        outputs.append(int(p_flag and n1_flag and n2_flag))
    inputs = [P,Q,R];outputs = torch.tensor(outputs)
    return inputs,outputs

tc_in,tc_out = make_testcase(B,N)
print(tc_out)
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


bgr = torch.zeros([B,N,N,1])

loss_func = torch.nn.BCELoss()

import matplotlib.pyplot as plt
plt.ion()
optim = torch.optim.Adam(net.parameters(),lr=2e-4)
for epoch in range(50):
    optim.zero_grad()
    p,_,_ = net(*tc_in)
    loss = loss_func(p.float().reshape([1,-1]),tc_out.float().reshape([1,-1]))
    loss.backward()
    optim.step()
    plt.cla()
    plt.scatter(range(p.shape[0]),p.detach())
    plt.pause(0.01)

print(p.reshape([-1]))
print(tc_out.reshape([-1]))

ttc_in,ttc_out = make_testcase(100,50)
p,_,_ = net(*ttc_in)
print("start the test case:")
print(ttc_out.reshape([-1]))
print(p.reshape([-1]).int())
plt.ioff()
plt.cla()
plt.scatter(range(ttc_out.shape[0]),ttc_out)
plt.scatter(range(p.shape[0]),p.detach())
print(loss_func(p.float().reshape([1,-1]),ttc_out.float().reshape([1,-1])))
plt.show()