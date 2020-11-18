from data_prepare import *
from Roberta_ft import *
from setting import *
import time
from seqeval.metrics import accuracy_score, precision_score, f1_score, classification_report

if __name__ == '__main__':
    # Roberta  ner-task fine-tuning
    # label type BIO, data set: ibo_char

    # training parameters
    parameters = TrainingParameters(vocab_file='../models/pretrained/Roberta/vocab.txt', output_dir='../output',
                                    train_file='../datasets/ner/ibo_char_val.txt', eval_file='../datasets/ner/ibo_char_val.txt',
                                    init_model_checkpoint='../models/pretrained/Roberta/pytorch_model.bin',
                                    model_config='../models/pretrained/Roberta/config.json',
                                    no_inputs_mask=True, no_logits=False, hidden_dropout_prob=0.1, task_type='CWS',
                                    hidden_size=1024)

    # loading model

    model = BertForTokenClassification(parameters.model_config, parameters)
    init_checkpoint = parameters.init_model_checkpoint
    state_dict_S = torch.load(init_checkpoint, map_location='cpu')
    #state_weight = {k[5:]:v for k,v in state_dict_S.items() if k.startswith('bert.')}
    #missing_keys,_ = model_S.bert.load_state_dict(state_weight,strict=False)
    missing_keys, unexpected_keys = model.load_state_dict(state_dict_S, strict=False)
    print("missing keys: ", missing_keys)
    print('unexpected_keys', unexpected_keys)

    tokenizer = BertTokenizer(vocab_file=parameters.vocab_file, do_lower_case=True)
    label2id = DataConvert.label_switcher(task_type=parameters.task_type)
    D = DataConvert(label2id_dict=label2id)
    # loading data
    train_examples, train_dataset = D.read_features(parameters.train_file, tokenizer=tokenizer,
                                                             max_seq_length=parameters.max_seq_length)

    eval_examples, eval_dataset = D.read_features(parameters.eval_file,tokenizer=tokenizer,
                                                           max_seq_length=parameters.max_seq_length)

    # allocate parameters
    train_sampler = RandomSampler(train_dataset)
    eval_sampler = RandomSampler(eval_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=parameters.train_batchsize,drop_last=True)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=parameters.eval_batchsize,drop_last=True)
    # learning rate decay strategy
    all_trainable_params = parameters.learning_decay_setting(model)
    # setting optimizer and scheduler
    scheduler_args, scheduler_class, optimizer, scheduler = parameters.auxiliary_training_setting(train_dataloader=train_dataloader,
                                                                                       all_trainable_params=all_trainable_params)

    # training start
    total_t0 = time.time()
    training_stats = []
    for epoch_i in range(0, parameters.training_epoches):

        # ========================================
        #               Training
        # ========================================

        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, parameters.training_epoches))
        print('Training...')

        # 统计单次 epoch 的训练时间
        t0 = time.time()

        # 重置每次 epoch 的训练总 loss
        total_train_loss = 0

        # 将模型设置为训练模式。这里并不是调用训练接口的意思
        model.train()

        # 训练集小批量迭代
        for step, batch in enumerate(train_dataloader):

            # 每经过40次迭代，就输出进度信息
            if step % 40 == 0 and not step == 0:
                elapsed = (time.time() - t0)
                print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

            # 准备输入数据，并将其拷贝到 gpu 中
            b_input_ids = batch[0]
            b_input_mask = batch[1]
            b_labels = batch[2]

            # 每次计算梯度前，都需要将梯度清 0，因为 pytorch 的梯度是累加的
            model.zero_grad()

            # 前向传播
            # 文档参见:
            # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
            # 该函数会根据不同的参数，会返回不同的值。 本例中, 会返回 loss 和 logits -- 模型的预测结果
            output = model(b_input_ids,
                                 token_type_ids=None,
                                 attention_mask=b_input_mask,
                                 labels=b_labels)

            # 累加 loss
            loss = output[0]
            total_train_loss += loss.item()

            # 反向传播
            loss.backward()

            # 梯度裁剪，避免出现梯度爆炸情况
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # 更新参数
            optimizer.step()

            # 更新学习率
            scheduler.step()

        # 平均训练误差
        avg_train_loss = total_train_loss / len(train_dataloader)

        # 单次 epoch 的训练时长
        training_time = (time.time() - t0)

        print("")
        print("  Average training loss: {0:.2f}".format(avg_train_loss))
        print("  Training epcoh took: {:}".format(training_time))

        # ========================================
        #               Validation
        # ========================================
        # 完成一次 epoch 训练后，就对该模型的性能进行验证

        print("")
        print("Running Validation...")

        t0 = time.time()

        # 设置模型为评估模式
        model.eval()

        # Tracking variables
        total_eval_accuracy = 0
        total_eval_loss = 0
        nb_eval_steps = 0

        # Evaluate data for one epoch
        for batch in eval_dataloader:
            # 将输入数据加载到 gpu 中
            b_input_ids = batch[0]
            b_input_mask = batch[1]
            b_labels = batch[2]
            # 评估的时候不需要更新参数、计算梯度
            with torch.no_grad():
                output = model(b_input_ids,
                                       token_type_ids=None,
                                       attention_mask=b_input_mask,
                                       labels=b_labels)
            # 累加 loss
            total_eval_loss += output[0].item()

            # 将预测结果和 labels 加载到 cpu 中计算
            logits = output[1].numpy()
            label_ids = b_labels.numpy()

            # 计算准确率
            total_eval_accuracy += classification_report(logits.argmax(dim=-1), label_ids)

        # 打印本次 epoch 的准确率
        avg_val_accuracy = total_eval_accuracy / len(eval_dataloader)
        print("  Accuracy: {0:.2f}".format(avg_val_accuracy))

        # 统计本次 epoch 的 loss
        avg_val_loss = total_eval_loss / len(eval_dataloader)

        # 统计本次评估的时长
        validation_time = (time.time() - t0)

        print("  Validation Loss: {0:.2f}".format(avg_val_loss))
        print("  Validation took: {:}".format(validation_time))

        # 记录本次 epoch 的所有统计信息
        training_stats.append(
            {
                'epoch': epoch_i + 1,
                'Training Loss': avg_train_loss,
                'Valid. Loss': avg_val_loss,
                'Valid. Accur.': avg_val_accuracy,
                'Training Time': training_time,
                'Validation Time': validation_time
            }
        )

    print("")
    print("Training complete!")
    print("Total training took {:} (h:mm:ss)".format((time.time() - total_t0)))


