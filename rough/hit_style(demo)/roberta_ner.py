import logging


import os
import torch
from hit_style.utils_ner import read_features, label2id_dict
from hit_style.utils import divide_parameters
from transformers import AdamW, get_constant_schedule_with_warmup, get_linear_schedule_with_warmup, BertTokenizer, get_constant_schedule, BertConfig
from hit_style.modeling import BertForTokenClassification, RobertaForTokenClassificationAdaptorTraining
from textbrewer import TrainingConfig,BasicTrainer
from torch.utils.data import DataLoader, RandomSampler
from functools import partial

from hit_style.train_eval import ddp_predict

def args_check(logger, args):
    if os.path.exists(args.output_dir) and os.listdir(args.output_dir):
        logger.warning("Output directory () already exists and is not empty.")
    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                            args.gradient_accumulation_steps))

    if not args.do_train and not args.do_predict:
        raise ValueError("At least one of `do_train` or `do_predict` must be True.")

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count() if not args.no_cuda else 0
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        torch.distributed.init_process_group(backend='nccl')
        logger.info("rank %d device %s n_gpu %d distributed training %r", torch.distributed.get_rank(), device, n_gpu, bool(args.local_rank != -1))
    args.n_gpu = n_gpu
    args.device = device
    return device, n_gpu

def main():
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s -  %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S',
    )
    logger = logging.getLogger("Main")
    forward_batch_size = int(20/ 1)


    bert_config_file_S = '../../cwsmodel/pretrained/Roberta/config.json'
    output_encoded_layers = 'true'
    max_seq_length = 160
    #load bert config
    bert_config_S = BertConfig.from_json_file(bert_config_file_S)
    bert_config_S.output_hidden_states = (output_encoded_layers=='true')
    bert_config_S.num_labels = len(label2id_dict)
    assert max_seq_length <= bert_config_S.max_position_embeddings

    #read data
    train_examples = None
    train_dataset = None
    eval_examples = None
    eval_dataset = None
    num_train_steps = None

    vocab_file = '../../cwsmodel/pretrained/Roberta/vocab.txt'

    tokenizer = BertTokenizer(vocab_file=vocab_file, do_lower_case=True)

    train_file = '../../cwsmodel/ibo_char_train.txt'
    predict_file = '../../cwsmodel/ibo_char_val.txt'

    train_examples,train_dataset = read_features(train_file, tokenizer=tokenizer, max_seq_length=max_seq_length)

    eval_examples,eval_dataset = read_features(predict_file,tokenizer=tokenizer, max_seq_length=max_seq_length)

    #Build Model and load checkpoint
    model_S = BertForTokenClassification(bert_config_S)
    #Load student

    init_checkpoint_S = './cwsmodel/pretrained/Roberta/pytorch_model.bin'
    assert init_checkpoint_S is not None



    state_dict_S = torch.load(init_checkpoint_S, map_location='cpu')
    #state_weight = {k[5:]:v for k,v in state_dict_S.items() if k.startswith('bert.')}
    #missing_keys,_ = model_S.bert.load_state_dict(state_weight,strict=False)
    w= [(name,value) for name,value in model_S.named_parameters()]
    n = [name for name,value in model_S.bert.named_parameters()]
    wwww = []
    for pair in w:
        if pair[1].requires_grad is not True:
            wwww.append(pair)
    missing_keys, unexpected_keys = model_S.load_state_dict(state_dict_S,strict=False)
    logger.info(f"missing keys:{missing_keys}")
    logger.info(f"unexpected keys:{unexpected_keys}")


    do_train = True
    lr_decay = 0.8
    learning_rate = 0.00001
    collected = []
    if do_train:
        #parameters
        if lr_decay is not None:
            outputs_params = list(model_S.classifier.named_parameters())
            outputs_params, names = divide_parameters(outputs_params, lr = learning_rate)
            collected.extend(names)

            pooler_params = list(model_S.bert.pooler.named_parameters())
            pooler_params, names = divide_parameters(pooler_params, lr=learning_rate)


            collected.extend(names)

            Roberta_params = []
            n_layers = len(model_S.bert.encoder.layer)
            assert n_layers==24
            for i,n in enumerate(reversed(range(n_layers))):
                encoder_params = list(model_S.bert.encoder.layer[n].named_parameters())
                lr = learning_rate * lr_decay**(i+1)
                a, b= divide_parameters(encoder_params, lr = lr)
                Roberta_params+=a
                collected.extend(b)
                logger.info(f"{i},{n},{lr}")
            embed_params = [(name,value) for name,value in model_S.bert.named_parameters() if 'embedding' in name]

            logger.info(f"{[name for name,value in embed_params]}")
            lr = learning_rate * lr_decay**(n_layers+1)
            a,b= divide_parameters( embed_params, lr = lr)
            Roberta_params += a
            collected.extend(b)
            logger.info(f"embed lr:{lr}")
            all_trainable_params = outputs_params + Roberta_params + pooler_params
            w = sum(map(lambda x:len(x['params']), all_trainable_params))
            assert sum(map(lambda x:len(x['params']), all_trainable_params))==len(list(model_S.parameters())),\
                (sum(map(lambda x:len(x['params']), all_trainable_params)), len(list(model_S.parameters())))
        else:
            params = list(model_S.named_parameters())
            all_trainable_params = divide_parameters(params, lr=learning_rate)
        logger.info("Length of all_trainable_params: %d", len(all_trainable_params))


        train_sampler = RandomSampler(train_dataset)
        train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=forward_batch_size,drop_last=True)
        num_train_epochs = 20
        gradient_accumulation_steps =1
        official_schedule = 'linear'
        warmup_proportion = 0.2
        num_train_steps = len(train_dataloader) // gradient_accumulation_steps * num_train_epochs
        optimizer = AdamW(all_trainable_params, lr=learning_rate, correct_bias = False)
        if official_schedule == 'const':
            scheduler_class = get_constant_schedule_with_warmup
            scheduler_args = {'num_warmup_steps':int(warmup_proportion*num_train_steps)}
            #scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=int(args.warmup_proportion*num_train_steps))
        elif official_schedule == 'linear':
            scheduler_class = get_linear_schedule_with_warmup
            scheduler_args = {'num_warmup_steps':int(warmup_proportion*num_train_steps), 'num_training_steps': num_train_steps}
            #scheduler = get_linear_schedule_with_warmup(optimizer,num_warmup_steps=int(args.warmup_proportion*num_train_steps), num_training_steps = num_train_steps)
        elif official_schedule == 'const_nowarmup':
            scheduler_class = get_constant_schedule
            scheduler_args = {}
        else:
            raise NotImplementedError


        logger.warning("***** Running training *****")


        ckpt_frequency = 2
        output_dir = '../../output'
        ########### TRAINING ###########
        train_config = TrainingConfig(
            gradient_accumulation_steps = gradient_accumulation_steps,
            ckpt_frequency = ckpt_frequency,
            log_dir = output_dir,
            output_dir = output_dir,
            device = 'cpu',
        )
        logger.info(f"{train_config}")

        distiller = BasicTrainer(train_config = train_config,
                                   model = model_S,
                                   adaptor = RobertaForTokenClassificationAdaptorTraining)

        # evluate the model in a single process in ddp_predict
        callback_func = partial(ddp_predict,
                eval_examples=eval_examples,
                eval_dataset=eval_dataset,
            )
        with distiller:
            distiller.train(optimizer, scheduler_class=scheduler_class,
                              scheduler_args=scheduler_args,
                              max_grad_norm = 1.0,
                              dataloader=train_dataloader,
                              num_epochs=num_train_epochs, callback=callback_func)

    do_predict = 1
    if not do_train and do_predict:
        res = ddp_predict(model_S,eval_examples,eval_dataset,step=0)
        print (res)




if __name__ == "__main__":
    main()