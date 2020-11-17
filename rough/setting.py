
import pickle
import os
from transformers import AdamW, get_constant_schedule_with_warmup, get_linear_schedule_with_warmup, BertTokenizer, get_constant_schedule, BertConfig
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
import logging
logger = logging.getLogger("utils")
logger.setLevel(logging.INFO)

def divide_parameters(named_parameters,lr=None):
    no_decay = ['bias', 'LayerNorm.bias','LayerNorm.weight']
    decay_parameters_names = list(zip(*[(p,n) for n,p in named_parameters if not any((di in n) for di in no_decay)]))
    no_decay_parameters_names = list(zip(*[(p,n) for n,p in named_parameters if any((di in n) for di in no_decay)]))
    param_group = []
    if len(decay_parameters_names)>0:
        decay_parameters, decay_names = decay_parameters_names
        #print ("decay:",decay_names)
        if lr is not None:
            decay_group = {'params':decay_parameters,   'weight_decay': 0.8, 'lr':lr}
        else:
            decay_group = {'params': decay_parameters, 'weight_decay': 0.8}
        param_group.append(decay_group)

    if len(no_decay_parameters_names)>0:
        no_decay_parameters, no_decay_names = no_decay_parameters_names
        #print ("no decay:", no_decay_names)
        if lr is not None:
            no_decay_group = {'params': no_decay_parameters, 'weight_decay': 0.0, 'lr': lr}
        else:
            no_decay_group = {'params': no_decay_parameters, 'weight_decay': 0.0}
        param_group.append(no_decay_group)
    collected = []
    collected.extend(decay_names)
    collected.extend(no_decay_names)
    assert len(param_group)>0
    return param_group, collected



# general training parameters
class TrainingParameters:
    def __init__(self, vocab_file, output_dir, train_file, eval_file, init_model_checkpoint,
                model_config, no_inputs_mask, no_logits,
                 **kwargs):
        self.learning_rate = 0.001
        self.training_epoches = 5
        self.train_batchsize = 20
        self.eval_batchsize = 20
        self.vocab_file = vocab_file
        self.output_dir = output_dir
        self.train_file = train_file
        self.eval_file = eval_file
        self.max_seq_length = 160
        self.do_lower_case = True
        self.warmup_proportion = 0.1
        self.verbose_logging = True
        self.no_cuda = True
        self.gradient_accumulation_steps = 1
        self.local_rank = -1
        self.fp16 = False
        self.random_seed = 10236347
        self.load_model_type = 'bert'
        self.weight_decay_rate = 0.01
        self.PRINT_EVERY = 200
        self.weight = 1.0
        self.ckpt_frequency = 2
        self.init_model_checkpoint = init_model_checkpoint
        self.model_config = BertConfig.from_json_file(model_config)
        self.no_inputs_mask = no_inputs_mask
        self.no_logits = no_logits
        self.output_encoded_layers = True
        self.output_attention_layers = True
        self.lr_decay = 0.0001
        self.official_schedule = 'linear'

        for key, value in kwargs.items():
            setattr(self, key, value)

    def learning_decay_setting(self, model):
        parameters_involved = []
        if self.lr_decay is not None:
            outputs_params = list(model.classifier.named_parameters())
            outputs_params, names = divide_parameters(outputs_params, lr=self.learning_rate)
            parameters_involved.extend(names)

            pooler_params = list(model.bert.pooler.named_parameters())
            pooler_params, names = divide_parameters(pooler_params, lr=self.learning_rate)

            parameters_involved.extend(names)

            Roberta_params = []
            n_layers = len(model.bert.encoder.layer)
            assert n_layers == 24
            for i, n in enumerate(reversed(range(n_layers))):
                encoder_params = list(model.bert.encoder.layer[n].named_parameters())
                lr = self.learning_rate * self.lr_decay ** (i + 1)
                a, b = divide_parameters(encoder_params, lr=lr)
                Roberta_params += a
                parameters_involved.extend(b)
                logger.info(f"{i},{n},{lr}")
            embed_params = [(name, value) for name, value in model.bert.named_parameters() if 'embedding' in name]
            logger.info(f"{[name for name, value in embed_params]}")
            lr = self.learning_rate * self.lr_decay ** (n_layers + 1)
            a, b = divide_parameters(embed_params, lr=lr)
            Roberta_params += a
            parameters_involved.extend(b)
            logger.info(f"embed lr:{lr}")
            all_trainable_params = outputs_params + Roberta_params + pooler_params
            w = sum(map(lambda x: len(x['params']), all_trainable_params))
            assert sum(map(lambda x: len(x['params']), all_trainable_params)) == len(list(model.parameters())), \
                (sum(map(lambda x: len(x['params']), all_trainable_params)), len(list(model.parameters())))
        else:
            params = list(model.named_parameters())
            all_trainable_params = divide_parameters(params, lr=self.learning_rate)
        return all_trainable_params

    def auxiliary_training_setting(self, train_dataloader, all_trainable_params):
        num_train_epochs = self.training_epoches
        gradient_accumulation_steps = self.gradient_accumulation_steps
        official_schedule = 'linear'
        warmup_proportion = 0.2
        num_train_steps = len(train_dataloader) // gradient_accumulation_steps * num_train_epochs
        optimizer = AdamW(all_trainable_params, lr=self.learning_rate, correct_bias=False)
        if official_schedule == 'const':
            scheduler_class = get_constant_schedule_with_warmup
            scheduler_args = {'num_warmup_steps': int(warmup_proportion * num_train_steps)}
            scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=int(self.warmup_proportion*num_train_steps))
            return scheduler_args, scheduler_class, optimizer, scheduler
        elif official_schedule == 'linear':
            scheduler_class = get_linear_schedule_with_warmup
            scheduler_args = {'num_warmup_steps': int(warmup_proportion * num_train_steps),
                              'num_training_steps': num_train_steps}
            scheduler = get_linear_schedule_with_warmup(optimizer,num_warmup_steps=int(self.warmup_proportion*num_train_steps), num_training_steps = num_train_steps)
            return scheduler_args, scheduler_class, optimizer, scheduler
        elif official_schedule == 'const_nowarmup':
            scheduler_class = get_constant_schedule
            scheduler_args = {}
            scheduler = get_constant_schedule_with_warmup(optimizer)
            return scheduler_args, scheduler_class, optimizer, scheduler
        else:
            raise NotImplementedError