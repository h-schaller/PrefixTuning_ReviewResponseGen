import os

if __name__ == '__main__':

    # hyperparameters to tune via grid search
    for prefix_length in [1, 10, 20, 50, 80, 100, 200, 300]:
        for learning_rate in [1e-05, 5e-05, 8e-05]:
            commandline = 'python train_bart.py ' \
                          '--tuning_mode prefixtune ' \
                          '--do_train yes ' \
                          '--bsz 14 ' \
                          '--epoch 30 ' \
                          '--gradient_accumulation_steps 3 ' \
                          '--mid_dim 800 ' \
                          '--preseqlen {} ' \
                          '--learning_rate {}' \
                          ''.format(prefix_length, learning_rate)
            os.system(commandline)