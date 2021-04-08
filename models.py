import json
import mlflow
import numpy as np
import torch
import torch.nn as nn
import transformers

from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertModel
from transformers.modeling_outputs import SequenceClassifierOutput
from utils import get_batches, sigmoid


class BertForMultiLabelSequenceClassification(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.pos_weight = torch.ones(
            self.config.num_labels, requires_grad=False
        )

        self.init_weights()

    def freeze_bert(self):
        for name, param in self.bert.named_parameters():
            param.requires_grad = False

    def set_pos_weight(self, weight):
        self.pos_weight = weight*self.pos_weight

    def set_label_weights(self, label_weights):
        with open(label_weights, "r") as f:
            weights = json.load(f)

        assert set(weights.keys()) == set(self.config.label2id.keys())

        for idx, tag in self.config.id2label.items():
            self.pos_weight[idx] = weights[tag]

        print("Set label weight array to:", self.pos_weight)


    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[0, ...,
            config.num_labels - 1]`. If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            self.pos_weight = self.pos_weight.to(logits.device)
            loss_fct = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)
            loss = loss_fct(logits, labels.double())

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class MLFlowBertClassificationModel(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        chkp = context.artifacts["model"]
        self.model = BertForMultiLabelSequenceClassification.from_pretrained(chkp)
        self.tokenizer = transformers.BertTokenizer.from_pretrained(chkp)

    def predict(self, context, model_input):
        predictions = []
        for batch in get_batches(list(model_input.text), size=2):
            input_ids = self.tokenizer(batch, truncation=True, padding=True, return_tensors="pt")["input_ids"]
            predictions.append(sigmoid(self.model(input_ids).logits.detach().numpy()))
        return np.concatenate(predictions)
