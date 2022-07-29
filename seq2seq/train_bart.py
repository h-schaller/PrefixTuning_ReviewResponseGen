import os, sys
import argparse
import ntpath

# example: python train_run.py keyword temp_keyword _
if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    gpus = 1

    parser = argparse.ArgumentParser(description='data2text E2E training args.')
    parser.add_argument('--mode', type=str, default='review_response', help='')
    parser.add_argument('--tuning_mode', type=str, default='prefixtune', help='')
    parser.add_argument('--optim_prefix', type=str, default='yes', help='')
    parser.add_argument('--preseqlen', type=int, default=5, help='')
    parser.add_argument('--prefix_mode', type=str, default='activation', help='')
    parser.add_argument('--format_mode', type=str, default='cat', help='')
    parser.add_argument('--data_dir', type=str, default=None, help='')

    parser.add_argument('--dir_name', type=str, default=None, help='')
    parser.add_argument('--notes', type=str, default=None, help='')
    parser.add_argument('--lowdata_token', type=str, default='summarize', help='')
    parser.add_argument('--use_lowdata_token', type=str, default='yes', help='')


    parser.add_argument('--parametrize_emb', type=str, default='MLP', help='')
    parser.add_argument('--adapter_design', type=int, default=1, help='')
    parser.add_argument('--top_layers', type=int, default=1, help='')

    parser.add_argument('--do_train', type=str, default='yes', help='')

    parser.add_argument('--fp16', type=str, default='no', help='')


    # training parameters.
    parser.add_argument('--use_dropout', type=str, default='no', help='')
    parser.add_argument('--seed', type=int, default=101, help='') # old is 42
    parser.add_argument('--bsz', type=int, default=10, help='')
    parser.add_argument('--use_big', type=str, default='no', help='')
    parser.add_argument('--epoch', type=int, default=5, help='')
    parser.add_argument('--max_steps', type=int, default=400, help='')
    parser.add_argument('--eval_steps', type=int, default=50, help='')
    parser.add_argument('--warmup_steps', type=int, default=100, help='')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help='')
    parser.add_argument('--learning_rate', type=float, default=5e-05, help='')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='')
    parser.add_argument('--dropout', type=float, default=0.0, help='')


    parser.add_argument('--label_smoothing', type=float, default=0.0, help='')
    parser.add_argument('--length_pen', type=float, default=1.0, help='')
    parser.add_argument('--mid_dim', type=int, default=512, help='')
    parser.add_argument('--use_deep', type=str, default='no', help='')

    parser.add_argument('--prefix_model_path', type=str, default=None, help='')
    parser.add_argument('--finetune_model_path', type=str, default=None, help='')
    parser.add_argument('--checkpoint_path', default=None, help='')

    args = parser.parse_args()

    assert args.optim_prefix in ['yes', 'no']
    if args.optim_prefix == 'yes':
        assert args.preseqlen is not None
    assert args.prefix_mode in ['embedding', 'activation']
    assert args.format_mode in ['cat', 'infix', 'peek', 'nopeek']
    assert args.tuning_mode in ['prefixtune', 'finetune', 'finetune-top', 'bothtune', 'adaptertune']
    if args.prefix_model_path is not None:
        load_prefix_model = True
    else:
        load_prefix_model = False

    assert args.mode in ['e2e', 'cnn_dm', 'webnlg', 'triples', 'xsum', 'xsum_news', 'xsum_news_sport',
                         'review_response', 'review_response_small']

    if args.mode == 'xsum':
        data_dir = 'xsum'
        folder_name = "xsum_models/"
        max_source_length = 1024
        max_target_length = 60
        val_max_target_length = 60
        test_max_target_length = 100

        xsum_app = ' --max_source_length {} --max_target_length {} --val_max_target_length {} ' \
                    '--test_max_target_length {} '.format(max_source_length, max_target_length,
                                                         val_max_target_length, test_max_target_length)

        if args.fp16 == 'yes':
            xsum_app += ' --fp16 --fp16_opt_level O1 '

        assert args.optim_prefix == 'yes'

    elif args.mode == 'review_response':
        # data_dir = '../data/review_response'
        if args.tuning_mode == 'finetune':
            folder_name = 'finetuned_models/'
        elif args.tuning_mode == 'prefixtune':
            folder_name = 'prefix_models/'
        # folder_name = "review_response_models/"
        max_source_length = 256
        max_target_length = 128
        val_max_target_length = 128
        test_max_target_length = 128

        review_response_app = ' --max_source_length {} --max_target_length {} --val_max_target_length {} ' \
                   '--test_max_target_length {} '.format(max_source_length, max_target_length,
                                                         val_max_target_length, test_max_target_length)

        if args.fp16 == 'yes':
            review_response_app += ' --fp16 --fp16_opt_level O1 '

    elif args.mode == 'review_response_small':
        # data_dir = '../data/review_response'
        if args.tuning_mode == 'finetune':
            folder_name = 'finetuned_models/'
        elif args.tuning_mode == 'prefixtune':
            folder_name = 'prefix_models/'
        # folder_name = "review_response_models/"
        max_source_length = 132
        max_target_length = 60
        val_max_target_length = 60
        test_max_target_length = 60

        review_response_small_app = ' --max_source_length {} --max_target_length {} --val_max_target_length {} ' \
                              '--test_max_target_length {} '.format(max_source_length, max_target_length,
                                                                    val_max_target_length, test_max_target_length)

        if args.fp16 == 'yes':
            review_response_small_app += ' --fp16 --fp16_opt_level O1 '

        assert args.optim_prefix == 'yes'

    if not os.path.isdir(folder_name):
        os.mkdir(folder_name)

    batch_size = args.gradient_accumulation_steps * args.bsz

    if args.dir_name is None:
        # Model_FILE = args.mode + '_' + args.tuning_mode + '_' + args.optim_prefix[:1] + '_' + str(args.preseqlen) + \
        #              '_' + args.prefix_mode[:3] + '_' + args.format_mode[:3] + '_' + \
        #              'b={}-'.format(batch_size) + 'e={}_'.format(args.epoch) + 'd={}_'.format(args.dropout) + \
        #              'l={}_'.format(args.label_smoothing) + 'lr={}_'.format(args.learning_rate) \
        #              + 'w={}_'.format(args.weight_decay) + 's={}'.format(args.seed) + '_d={}'.format(args.use_deep[:1]) +\
        #              '_m={}'.format(args.mid_dim)
        if args.tuning_mode == 'prefixtune':
            norm_data_dir = ntpath.normpath(args.data_dir)
            restaurant = norm_data_dir.split("\\")[-1]
            Model_FILE = restaurant + '/' + args.tuning_mode + '_prelen=' + str(args.preseqlen) + '_lr={}'.format(args.learning_rate) + '_bs={}'.format(args.bsz)
            if not os.path.isdir(folder_name + restaurant):
                os.mkdir(folder_name + restaurant)

        elif args.tuning_mode == 'finetune':
            Model_FILE = args.tuning_mode + '_lr={}'.format(args.learning_rate) + '_bs={}'.format(args.bsz)
    else:
        Model_FILE = args.dir_name

    if args.notes is not None:
        Model_FILE += '_{}'.format(args.notes)

    logging_dir = os.path.join(folder_name, 'runs', Model_FILE)
    Model_FILE = '{}{}'.format(folder_name, Model_FILE)
    print("Model_FILE", Model_FILE)

    OLD_MODEL = 'facebook/bart-large'

    app = "--optim_prefix {} --preseqlen {} --prefix_mode {} --format_mode {} " \
          "--gradient_accumulation_steps {} --learning_rate {} --weight_decay {} --seed {} " \
          "--mid_dim {} --use_dropout {} --prefix_dropout {} ".\
        format(args.optim_prefix, args.preseqlen, args.prefix_mode, args.format_mode,
               args.gradient_accumulation_steps, args.learning_rate, args.weight_decay, args.seed,
               args.mid_dim, args.use_dropout, args.dropout)

    if args.prefix_mode == 'embedding':
        app += ' --parametrize_emb {} '.format(args.parametrize_emb)

    if args.tuning_mode == 'adaptertune':
        app += ' --adapter_design {} '.format(args.adapter_design)

    if args.mode == 'xsum' or args.mode == 'xsum_news' or args.mode == 'xsum_news_sport': 
        app += xsum_app

    if args.mode == 'review_response':
        app += review_response_app

    if args.mode == 'review_response_small':
        app += review_response_small_app

    if args.tuning_mode == 'finetune-top':
        app += ' --top_layers {} '.format(args.top_layers)

    controlprefix = ('yes' if args.tuning_mode == 'prefixtune' else 'no')

    if args.do_train == 'yes':
        COMMANDLINE = 'python finetune.py ' \
                      '--model_name_or_path {} ' \
                      '--output_dir {} ' \
                      '--data_dir {} ' \
                      '--tuning_mode {} ' \
                      '--preseqlen {} ' \
                      '--do_train ' \
                      '--label_smoothing {} ' \
                      '--use_deep {} ' \
                      '--gpus {} ' \
                      '--train_batch_size {} ' \
                      '--eval_batch_size {} ' \
                      '--num_train_epochs {} ' \
                      '--checkpoint_path {} '.format(OLD_MODEL, Model_FILE, args.data_dir, args.tuning_mode, args.preseqlen, args.label_smoothing, args.use_deep,
                                                      gpus, args.bsz, args.bsz - 2, args.epoch, args.checkpoint_path)
    else:
        if args.tuning_mode == 'finetune':
            assert args.finetune_model_path is not None
            print('loading from the finetune model {}'.format(args.finetune_model_path))
            # Model_FILE = args.finetune_model_path + '_decode_eval' + '_{}'.format(args.length_pen)
            # print('writing the decoded results to {}'.format(Model_FILE))
            COMMANDLINE = 'python finetune.py ' \
                          '--model_name_or_path {} ' \
                          '--output_dir {} ' \
                          '--data_dir {} ' \
                          '--tuning_mode {} ' \
                          '--preseqlen {} ' \
                          '--do_predict ' \
                          '--use_deep {} ' \
                          '--gpus {} ' \
                          '--train_batch_size {} ' \
                          '--eval_batch_size {} ' \
                          '--length_penalty {} ' \
                          '--num_train_epochs {} '.format(args.finetune_model_path, Model_FILE, args.data_dir,
                                                          args.tuning_mode, args.preseqlen,  args.use_deep,
                                                          gpus, args.bsz,  args.bsz - 2, args.length_pen, args.epoch)
        else:
            assert args.prefix_model_path is not None
            print('loading from the prefix model {}'.format(args.prefix_model_path))
            print('loading from the main model {}'.format(OLD_MODEL))
            #Model_FILE = args.prefix_model_path + '_decode_eval' + '_{}'.format(args.length_pen)
            #print('writing the decoded results to {}'.format(Model_FILE))
            COMMANDLINE = 'python finetune.py ' \
                          '--model_name_or_path {} ' \
                          '--prefixModel_name_or_path {} ' \
                          '--output_dir {} ' \
                          '--data_dir {} ' \
                          '--tuning_mode {} ' \
                          '--preseqlen {} ' \
                          '--do_predict ' \
                          '--use_deep {} ' \
                          '--gpus {} ' \
                          '--train_batch_size {} ' \
                          '--eval_batch_size {} ' \
                          '--seed {} ' \
                          '--length_penalty {} ' \
                          '--num_train_epochs {} '.format(OLD_MODEL, args.prefix_model_path, Model_FILE, args.data_dir,
                                                          args.tuning_mode, args.preseqlen, args.use_deep, gpus,
                                                          args.bsz,  args.bsz - 2, args.seed, args.length_pen, args.epoch)

    COMMANDLINE += app

    with open(Model_FILE + '.sh', 'w') as f:
        print(COMMANDLINE, file=f)

    print("COMMANDLINE", COMMANDLINE)
    os.system(COMMANDLINE)
