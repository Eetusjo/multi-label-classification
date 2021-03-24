import torch.nn as nn

from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertModel
from transformers.modeling_outputs import SequenceClassifierOutput


class BertForMultiLabelSequenceClassification(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()

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
            loss_fct = nn.BCEWithLogitsLoss()
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


if __name__ == "__main__":
    import transformers
    import datasets
    import torch
    import json

    from utils import fincore_to_dict_upper, fincore_tags_to_onehot

    train = fincore_to_dict_upper("../../data/fincore-train.tsv", "train")

    with open("fincore.train.jsonl", "w") as f:
        for sample in train:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    dataset = datasets.load_dataset(
        'json', data_files={"train": "fincore.train.jsonl"}
    )

    tokenizer = transformers.BertTokenizer.from_pretrained("TurkuNLP/bert-base-finnish-cased-v1")

    def preprocess_function(examples):
        preproc = tokenizer(examples["text"], truncation=True)
        preproc["labels"] = fincore_tags_to_onehot(examples["tags"])
        return preproc

    encoded_dataset = dataset.map(preprocess_function, batched=True)

    sample = encoded_dataset["train"][0]

    print(sample)

    raise ValueError()

    model = BertForMultiLabelSequenceClassification.from_pretrained(
        "TurkuNLP/bert-base-finnish-cased-v1", num_labels=5
    )

    print(
        model(torch.tensor(sample["input_ids"]).view(1, -1))
    )
