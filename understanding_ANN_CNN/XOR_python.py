#%%
input1 = float(input("enter first input  : "))
input2 = float(input("enter second input : "))

# set weights and biases
w1 = [1.0, -1.0]
bias1 = 0.5
w2 = [-1.0, 1.0]
bias2 = 0.5
w3 = [1.0, 1.0]
bias3 = -1.5

# compute outputs for first two neurons
dot1 = input1 * w1[0] + input2 * w1[1] + bias1
out1 = 1.0 if dot1 > 0.5 else 0.0
dot2 = input1 * w2[0] + input2 * w2[1] + bias2
out2 = 1.0 if dot2 > 0.5 else 0.0

# compute output for third neuron
dot3 = out1 * w3[0] + out2 * w3[1] + bias3
out3 = 1.0 if dot3 > 0.5 else 0.0

print(out3)
# %%
