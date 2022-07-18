import os

if __name__ == '__main__':

    # hyperparameters to tune via grid search
    for learning_rate in [1e-05, 5e-05, 8e-05]:
        commandline = 'python train_bart.py ' \
                      '--tuning_mode finetune ' \
                      '--do_train yes ' \
                      '--bsz 5 ' \
                      '--epoch 30 ' \
                      '--gradient_accumulation_steps 3 ' \
                      '--mid_dim 800 ' \
                      '--learning_rate {}' \
                      ''.format(learning_rate)
        os.system(commandline)
