from torch import nn
from transformers import BertModel, BertConfig
from transformers.modeling_bert import BertPreTrainedModel


class BertForTokenClassification(BertPreTrainedModel):
    def __init__(self, config, config_extra):
        super().__init__(config)
        self.config = config
        self.task_type = config_extra.task_type
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config_extra.hidden_dropout_prob)
        self.classifier = nn.Linear(config_extra.hidden_size, config.num_labels)
        self.init_weights()

    def forward(self, input_ids, attention_mask=None,labels=None, token_type_ids=None,
        position_ids=None, head_mask=None, inputs_embeds=None):
        # 直接取得bert的输出结果即经过了 embedding -> encoder -> pooler 的结果
        # 这里应包括了 最后一层的 output(batch_size, max_sentecne_seq),
        # 还有经过了dense层的logits 还有weights of attention layers
        transformers_hidden_states = self.bert(
            input_ids, attention_mask, token_type_ids, position_ids, head_mask, inputs_embeds
        )
        # 选取最后一层的output,先经过dropout以后再经过最后的分类尾巴层
        transformers_sequence_output = transformers_hidden_states[0]
        transformers_sequence_output = self.dropout(transformers_sequence_output)
        logits = self.classifier(transformers_sequence_output)

        output = (logits,)
        # 计算交叉熵
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.config.num_labels)[active_loss]
                active_labels = labels.view(-1)[active_loss]
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

            output = (loss,) + output



        return output  # (loss), scores, (hidden_states), (attentions)

    def predict(self, input_ids, attention_mask=None, labels=None, token_type_ids=None,
                position_ids=None, head_mask=None, inputs_embeds=None):
        transformers_hidden_states = self.bert(
            input_ids, attention_mask, token_type_ids, position_ids, head_mask, inputs_embeds
        )
        # 选取最后一层的output,先经过dropout以后再经过最后的分类尾巴层
        transformers_sequence_output = transformers_hidden_states[0]
        transformers_sequence_output = self.dropout(transformers_sequence_output)
        logits = self.classifier(transformers_sequence_output)

        labels_predicted = logits.numpy().argmax(dim=-1)

        return labels_predicted


