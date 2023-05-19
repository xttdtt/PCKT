"""Here are some fixed parameters"""
# the maximum number of program training
epochs = 1000
# batch size of samples
bs = 256
# early-stop mechanism
early_stop = 20
# dropout ratio
keep_rate = 0.5
# ratio of training set and test set
split_rate = 0.7

"""
Here are some non-fixed parameters
When performing ablation experiments, you can modify the values here
"""
# this program only supports two datasets:Assist09 and Assist12
dataset = "Assist09"
# learning rate
lr = 0.01
# dimensions of embedding matrix
embed_dim = 512
