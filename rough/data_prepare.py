import os, pickle
import torch
from torch.utils.data import TensorDataset
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

label2id_dict = {
    'O': 0,
    'B-LOC': 1,
    'I-LOC': 2,
    'B-ORG': 3,
    'I-ORG': 4,
    'B-PER': 5,
    'I-PER': 6
}

id2label_dict = {
    0: 'O',
    1: 'B-LOC',
    2: 'I-LOC',
    3: 'B-ORG',
    4: 'I-ORG',
    5: 'B-PER',
    6: 'I-PER'
}


class Examples:
    def __init__(self, tokens, label_ids):
        self.tokens = tokens
        self.label_ids = label_ids

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        s = ""
        s += f"tokens: {''.join(self.tokens)}\n"
        s += f"labels: {' '.join(str(i) for i in self.label_ids)}\n"
        return s


class Features:
    def __init__(self, token_ids, input_mask, label_ids):
        self.token_ids = token_ids
        self.input_mask = input_mask
        self.label_ids = label_ids

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        s = ""
        s += f"token_ids: {' '.join(str(i) for i in  self.token_ids)}\n"
        s += f"label_ids: {' '.join(str(i) for i in self.label_ids)}\n"
        s += f"input_mask:{' '.join(str(i) for i in self.input_mask)}\n"
        return s


class FeaturesGroup:
    def __init__(self, features: [Features], label2id: dict, id2label: dict, task_type: str, name: str):
        self.features = features
        self.label2id = label2id
        self.id2label = id2label
        self.task_type = task_type
        self.name = name

    def __str__(self):
        num_of_sentences = len(self.features)
        label_type = self.label2id
        return "instances numbre: " + str(num_of_sentences) + " label type: " + str(label_type.keys())

    def __repr__(self):
        k = []
        for f in self.features:
            s = ""
            s += f"token_ids: {' '.join(str(i) for i in  f.token_ids)}\n"
            s += f"label_ids: {' '.join(str(i) for i in f.label_ids)}\n"
            s += f"input_mask:{' '.join(str(i) for i in f.input_mask)}\n"
            k.append(s)
        return ' '.join(k)
# 定义一个类，主要是list of Example
# 但需要包括labelid_dict和id2dict


#  该类需要有如下功能
#  1. 读取原始数据统一成 token\tlabel 保存
#  2. 用dataloader方便深度模型处理
#  3. label交叉标, 交叉标以后得分别存储
class DataConvert:
    @staticmethod
    def relabel(models, features_of_dataset, examples_of_dataset, batch_size=10):
        for model in models:
            for dataset, dataset_example in zip(features_of_dataset, examples_of_dataset):
                new_dataset_example = dataset_example
                # 若其不是一个标签体系
                if dataset.task_type != model.task_type:
                    predict_sampler = SequentialSampler(dataset)
                    predict_dataloader = DataLoader(dataset, sampler=predict_sampler, batch_size=batch_size, drop_last=True)
                    for step, batch in enumerate(predict_dataloader):
                        labelid_predict = []
                        with torch.no_grad():
                            b_input_ids = batch[0]
                            b_input_mask = batch[1]
                            b_labels = batch[2]

                            labels_predicted = model.predict(b_input_ids,
                                       token_type_ids=None,
                                       attention_mask=b_input_mask,
                                       labels=b_labels)
                            labelid_predict.append(labels_predicted)
                        for ind in range(len(labelid_predict)):
                            new_dataset_example[ind+step*batch_size].label_ids = labelid_predict[ind]
                new_dataset_feature = DataConvert.example_to_features_easy(new_dataset_example)
                with open(model.task_type + '_' + dataset.name, 'wb') as f:
                    pickle.dump([new_dataset_example, new_dataset_feature], f)

    @staticmethod
    def read_examples(input_file):
        examples = []
        tokens = []
        label_ids = []
        errors = 0
        with open(input_file, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                if len(line.strip()) == 0:
                    if len(tokens) > 0:
                        examples.append(Examples(tokens, label_ids))
                        tokens = []
                        label_ids = []
                    continue
                try:
                    token, label = line.strip().split('\t')
                except ValueError:
                    errors += 1
                    continue
                tokens.append(token)
                label_ids.append(label2id_dict[label])
            if len(tokens) > 0:
                examples.append(Examples(tokens, label_ids))
        print("Num errors: ", errors)
        return examples

    @staticmethod
    def example_to_features_easy(examples, cls_token='[CLS]', sep_token='[SEP]', pad_token_id=0):
        features = []
        pad_label = [label2id_dict['O']]
        for example in examples:
            tokens = [cls_token] + example.tokens[:max_seq_length - 2] + [sep_token]
            label_ids = pad_label + example.label_ids[:max_seq_length - 2] + pad_label

            token_ids = tokenizer.convert_tokens_to_ids(tokens)
            input_mask = [1] * len(token_ids)

            padding_length = max_seq_length - len(token_ids)
            token_ids = token_ids + [pad_token_id] * padding_length
            input_mask = input_mask + [0] * padding_length
            label_ids = label_ids + pad_label * padding_length

            assert len(token_ids) == len(input_mask) == len(label_ids)

            features.append(Features(token_ids=token_ids, input_mask=input_mask, label_ids=label_ids))

        return features


    @staticmethod
    def convert_example_to_features(input_file, tokenizer, max_seq_length,
                                    cls_token='[CLS]', sep_token='[SEP]', pad_token_id=0):
        features = []

        examples = DataConvert.read_examples(input_file)

        # convert token to ids
        pad_label = [label2id_dict['O']]
        for example in examples:
            tokens = [cls_token] + example.tokens[:max_seq_length - 2] + [sep_token]
            label_ids = pad_label + example.label_ids[:max_seq_length - 2] + pad_label

            token_ids = tokenizer.convert_tokens_to_ids(tokens)
            input_mask = [1] * len(token_ids)

            padding_length = max_seq_length - len(token_ids)
            token_ids = token_ids + [pad_token_id] * padding_length
            input_mask = input_mask + [0] * padding_length
            label_ids = label_ids + pad_label * padding_length

            assert len(token_ids) == len(input_mask) == len(label_ids)

            features.append(Features(token_ids=token_ids, input_mask=input_mask, label_ids=label_ids))

        return examples, features

    @staticmethod
    def read_features(input_file, max_seq_length=160, tokenizer=None, cls_token='[CLS]', sep_token='[SEP]',
                      pad_token_id=0):
        cached_features_file = input_file + f'.cached_feat_{max_seq_length}'
        if os.path.exists(cached_features_file):
            with open(cached_features_file, 'rb') as f:
                examples, features = pickle.load(f)
        else:
            examples, features = DataConvert.convert_example_to_features(input_file, tokenizer, max_seq_length,
                                                                         cls_token,
                                                                         sep_token, pad_token_id)
            with open(cached_features_file, 'wb') as f:
                pickle.dump([examples, features], f)

        all_token_ids = torch.tensor([f.token_ids for f in features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_ids for f in features], dtype=torch.long)

        dataset = TensorDataset(all_token_ids, all_input_mask, all_label_ids)

        return examples, dataset


if __name__ == '__main__':
    from transformers import BertTokenizer
    tokenizer = BertTokenizer(vocab_file='./models/pretrained/Roberta/vocab.txt')

    input_file = './models/ibo_char_train.txt'
    max_seq_length = 128

    dataset = DataConvert.read_features(input_file, 128,tokenizer)
    #print (f"length of datasets: {len(datasets)}")
    #print (datasets[0])
    #print (datasets[-1])

    examples = DataConvert.read_examples(input_file)
    length = [len(example.tokens) for example in examples]
    import numpy as np
    print (np.max(length),np.mean(length),np.percentile(length,99))
    print (sum(i>160 for i in length)/len(length))







