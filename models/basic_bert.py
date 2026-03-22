"""
Inspired by the setup of the final project from course I recently took 
(https://web.stanford.edu/class/cs224n/), we are skinning the BERT model
and recreating it from the ground up, based on the model definition from
Hugging Face's transformers library, while still using the pretrained 
weights. This will allow us more control in case we decide to customize 
the base model.
"""
import torch
import transformers
from einops import rearrange
from torch import nn
from transformers import (
    BertConfig,
    BertTokenizer,
    BertModel
)
from typing import (
    Any, 
    List, 
    Union, 
    Tuple
)


class CustomBertEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.word_embeddings = nn.Embedding(
            config.vocab_size, 
            config.hidden_size, 
            padding_idx=config.pad_token_id
        )
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings,
            config.hidden_size
        )
        self.token_type_embeddings = nn.Embedding(
            config.type_vocab_size,
            config.hidden_size
        )
        self.embedding_dropout = nn.Dropout(config.hidden_dropout_prob)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        position_ids = torch.arange(config.max_position_embeddings).unsqueeze(0)
        self.register_buffer('position_ids', position_ids)

    def forward(self, input_ids, token_type_ids):
        sequence_length = input_ids.shape[1]
        position_ids = self.position_ids[:, :sequence_length]

        input_embeddings = self.word_embeddings(input_ids)
        input_embeddings += self.position_embeddings(position_ids) 
        input_embeddings += self.token_type_embeddings(token_type_ids)
        input_embeddings = self.layer_norm(input_embeddings)
        input_embeddings = self.embedding_dropout(input_embeddings)

        return input_embeddings
        

class CustomBertLayer(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.num_attention_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size

        self.query_linear = nn.Linear(config.hidden_size, config.hidden_size, bias=True)
        self.key_linear = nn.Linear(config.hidden_size, config.hidden_size, bias=True)
        self.value_linear = nn.Linear(config.hidden_size, config.hidden_size, bias=True)
        self.attention_dropout = nn.Dropout(p=config.attention_probs_dropout_prob)

        self.attention_out_dense = nn.Linear(config.hidden_size, config.hidden_size, bias=True)
        self.attention_out_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.attention_out_dropout = nn.Dropout(p=config.hidden_dropout_prob)

        self.intermediate_dense = nn.Linear(config.hidden_size, config.intermediate_size, bias=True)
        self.intermediate_activation = nn.functional.gelu

        self.out_dense = nn.Linear(config.intermediate_size, config.hidden_size, bias=True)
        self.out_layer_norm = nn.LayerNorm(config.hidden_size)
        self.out_dropout = nn.Dropout(config.hidden_dropout_prob)
    
    def attention(self, embedded_input, attention_mask):
        query = self.query_linear(embedded_input)
        key = self.key_linear(embedded_input)
        value = self.value_linear(embedded_input)

        query = rearrange(query, "b t (h d) -> b h t d", h=self.num_attention_heads)
        key = rearrange(key, "b t (h d) -> b h t d", h=self.num_attention_heads)
        value = rearrange(value, "b t (h d) -> b h t d", h=self.num_attention_heads)

        O = nn.functional.scaled_dot_product_attention(
            query, key, value, 
            scale=int(self.hidden_size/self.num_attention_heads) ** (-1/2),
            attn_mask=((1 - attention_mask[:, None, None, :]) * -10 ** 5).float(),
            is_causal=False
        )
        O = rearrange(O, "b h t d -> b t (h d)")
        O = self.attention_dropout(O)
        
        return O

    def post_attention(self, attention_input, attention_output):
        attention_output = self.attention_out_dense(attention_output)
        attention_output = self.attention_out_dropout(attention_output)
        attention_output = self.attention_out_layer_norm(attention_output + attention_input)

        return attention_output
    
    def ffn(self, input):
        output = self.intermediate_dense(input)
        output = self.intermediate_activation(output)
        output = self.out_dense(output)
        output = self.out_dropout(output)
        output = self.out_layer_norm(output + input)

        return output
    
    def forward(self, embedded_input, attention_mask):
        output = self.attention(embedded_input, attention_mask)
        output = self.post_attention(embedded_input, output)
        output = self.ffn(output)

        return output
    

class CustomBertPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.pooler_linear = nn.Linear(config.hidden_size, config.hidden_size)
        self.tanh = nn.functional.tanh

    def forward(self, x):
        x = self.pooler_linear(x)
        x = self.tanh(x)

        return x


class CustomBertModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.embedding_layer = CustomBertEmbeddings(config)
        self.attention_layers = nn.ModuleList([CustomBertLayer(config) for i in range(config.num_hidden_layers)])
        self.pooler_layer = CustomBertPooler(config)

    def forward(self, input_ids, token_type_ids, attention_mask):
        output = self.embedding_layer(input_ids, token_type_ids)
        for i, attention_layer in enumerate(self.attention_layers):
            output = attention_layer(output, attention_mask)
        pooler_output = self.pooler_layer(output[:, 0, :]) # Only pass the CLS token output to the pooler

        return {"last_hidden_state": output, "pooler_output": pooler_output}
    
    def hidden_state_to_token(self, hidden_state):
        return torch.einsum("b t d, v d -> b t v", hidden_state, self.embedding_layer.word_embeddings.weight)
    
    def from_pretrained(checkpoint="google-bert/bert-base-multilingual-cased", config=None):
        hf_bert_model = BertModel.from_pretrained(checkpoint).eval()
        config = hf_bert_model.config if config is None else config
        custom_bert_model = CustomBertModel(config).eval()

        # Overwrite embedding layer weights
        custom_bert_model.embedding_layer.word_embeddings.load_state_dict(
            hf_bert_model.embeddings.word_embeddings.state_dict()
        )
        custom_bert_model.embedding_layer.position_embeddings.load_state_dict(
            hf_bert_model.embeddings.position_embeddings.state_dict()
        )
        custom_bert_model.embedding_layer.token_type_embeddings.load_state_dict(
            hf_bert_model.embeddings.token_type_embeddings.state_dict()
        )
        custom_bert_model.embedding_layer.layer_norm.weight.data = hf_bert_model.state_dict()[
            "embeddings.LayerNorm.weight"
        ]
        custom_bert_model.embedding_layer.layer_norm.bias.data = hf_bert_model.state_dict()[
            "embeddings.LayerNorm.bias"
        ]

        # Overwrite bert layer weights
        for i in range(config.num_hidden_layers):

            # Overwrite Q, K, V projection weights and biases
            custom_bert_model.attention_layers[i].query_linear.weight.data = hf_bert_model.state_dict()[f"encoder.layer.{i}.attention.self.query.weight"]
            custom_bert_model.attention_layers[i].query_linear.bias.data = hf_bert_model.state_dict()[f"encoder.layer.{i}.attention.self.query.bias"]
            custom_bert_model.attention_layers[i].key_linear.weight.data = hf_bert_model.state_dict()[f"encoder.layer.{i}.attention.self.key.weight"]
            custom_bert_model.attention_layers[i].key_linear.bias.data = hf_bert_model.state_dict()[f"encoder.layer.{i}.attention.self.key.bias"]
            custom_bert_model.attention_layers[i].value_linear.weight.data = hf_bert_model.state_dict()[f"encoder.layer.{i}.attention.self.value.weight"]
            custom_bert_model.attention_layers[i].value_linear.bias.data = hf_bert_model.state_dict()[f"encoder.layer.{i}.attention.self.value.bias"]
            
            # Overwrite attention out weights and biases
            custom_bert_model.attention_layers[i].attention_out_dense.weight.data = hf_bert_model.state_dict()[f"encoder.layer.{i}.attention.output.dense.weight"]
            custom_bert_model.attention_layers[i].attention_out_dense.bias.data = hf_bert_model.state_dict()[f"encoder.layer.{i}.attention.output.dense.bias"]

            # Overwrite attention out layer norm weights and biases
            custom_bert_model.attention_layers[i].attention_out_layer_norm.weight.data = hf_bert_model.state_dict()[f"encoder.layer.{i}.attention.output.LayerNorm.weight"]
            custom_bert_model.attention_layers[i].attention_out_layer_norm.bias.data = hf_bert_model.state_dict()[f"encoder.layer.{i}.attention.output.LayerNorm.bias"]

            # Overwrite intermediate dense weights and biases
            custom_bert_model.attention_layers[i].intermediate_dense.weight.data = hf_bert_model.state_dict()[f"encoder.layer.{i}.intermediate.dense.weight"]
            custom_bert_model.attention_layers[i].intermediate_dense.bias.data = hf_bert_model.state_dict()[f"encoder.layer.{i}.intermediate.dense.bias"]

            # Overwrite out dense weights and biases
            custom_bert_model.attention_layers[i].out_dense.weight.data = hf_bert_model.state_dict()[f"encoder.layer.{i}.output.dense.weight"]
            custom_bert_model.attention_layers[i].out_dense.bias.data = hf_bert_model.state_dict()[f"encoder.layer.{i}.output.dense.bias"]

            # Overwrite out layer norm weights and biases
            custom_bert_model.attention_layers[i].out_layer_norm.weight.data = hf_bert_model.state_dict()[f"encoder.layer.{i}.output.LayerNorm.weight"]
            custom_bert_model.attention_layers[i].out_layer_norm.bias.data = hf_bert_model.state_dict()[f"encoder.layer.{i}.output.LayerNorm.bias"]

        # Overwrite pooler layer weights andbiases
        custom_bert_model.pooler_layer.pooler_linear.weight.data = hf_bert_model.state_dict()[f"pooler.dense.weight"]
        custom_bert_model.pooler_layer.pooler_linear.bias.data = hf_bert_model.state_dict()[f"pooler.dense.bias"]

        return custom_bert_model


def custom_bert_unit_test():

    # Test functionality of custom base bert model to ensure that
    # the custom model's outputs are exactly equal to those of the 
    # hf model. 

    sentence = """
    ΒΙΒΛΟΣ γενέσεως Ἰησοῦ Χριστοῦ υἱοῦ Δαυεὶδ υἱοῦ Ἀβρααμ.
    """
    tokenizer = BertTokenizer.from_pretrained("google-bert/bert-base-multilingual-cased")
    input = tokenizer(sentence, return_tensors="pt")

    hf_bert_model = BertModel.from_pretrained("google-bert/bert-base-multilingual-cased")
    custom_bert_model = CustomBertModel.from_pretrained()

    # Test for exactness of embedding layer output
    hf_bert_embeddings = hf_bert_model.embeddings(input["input_ids"], input["token_type_ids"])
    custom_bert_embeddings = custom_bert_model.embedding_layer(input["input_ids"], input["token_type_ids"])
    
    assert torch.allclose(hf_bert_embeddings, custom_bert_embeddings, rtol=1e-4, atol=1e-4), "The embedding output of the custom model is not accurate."
    print(f"The embedding layer outputs of the custom Bert model are exactly correct.")

    # Test for exactness of self attention
    # For simplicity, this test will only be conducted on the first attention layer
    hf_model_self_attention = hf_bert_model.encoder.layer[0].attention.self(hf_bert_embeddings)[0]
    custom_model_self_attention = custom_bert_model.attention_layers[0].attention(hf_bert_embeddings, input["attention_mask"])

    assert torch.allclose(hf_model_self_attention, custom_model_self_attention, rtol=1e-4, atol=1e-4), "The self attention output of the custom model is not accurate."
    print(f"The self attention outputs of the custom Bert model are exactly correct.")

    # Test for exactness of final output
    hf_model_output = hf_bert_model(**input)
    custom_model_output = custom_bert_model(input["input_ids"], input["token_type_ids"], input["attention_mask"])
    
    assert torch.allclose(custom_model_output["last_hidden_state"], hf_model_output["last_hidden_state"], rtol=1e-4, atol=1e-4), "The last hidden state output of the custom model is not accurate."
    print(f"The last hidden state outputs of the custom Bert model are exactly correct.")
    assert torch.allclose(custom_model_output["pooler_output"], hf_model_output["pooler_output"], rtol=1e-4, atol=1e-4), "The pooler output of the custom model is not accurate."
    print(f"The pooler outputs of the custom Bert model are exactly correct.")


if __name__ == "__main__":

    custom_bert_unit_test()