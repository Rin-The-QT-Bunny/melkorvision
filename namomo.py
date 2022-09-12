from scapy.all import *
import time

for i in range(1000):
    sendp(Ether(dst="ff:ff:ff:ff:ff:ff"))