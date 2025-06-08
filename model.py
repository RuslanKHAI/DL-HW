import torch
from typing import Type
from torch import nn
from dataset import TextDataset


class LanguageModel(nn.Module):
    def __init__(self, dataset: TextDataset, embed_size: int = 256, hidden_size: int = 256,
                 rnn_type: Type = nn.RNN, rnn_layers: int = 1):
        """
        Model for text generation
        :param dataset: text data dataset (to extract vocab_size and max_length)
        :param embed_size: dimensionality of embeddings
        :param hidden_size: dimensionality of hidden state
        :param rnn_type: type of RNN layer (nn.RNN or nn.LSTM)
        :param rnn_layers: number of layers in RNN
        """
        super(LanguageModel, self).__init__()
        self.dataset = dataset  # required for decoding during inference
        self.vocab_size = dataset.vocab_size
        self.max_length = dataset.max_length
        self.hidden_size = hidden_size
        self.rnn_type = rnn_type
        self.rnn_layers = rnn_layers

        """
        YOUR CODE HERE (⊃｡•́‿•̀｡)⊃━✿✿✿✿✿✿
        Create necessary layers
        """

        self.embedding = nn.Embedding(self.vocab_size, embed_size)
        self.rnn = rnn_type(
            input_size=embed_size,
            hidden_size=hidden_size,
            num_layers=rnn_layers,
            batch_first=True
        )
        self.linear = nn.Linear(hidden_size, self.vocab_size)

    def forward(self, indices: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        Compute forward pass through the model and
        return logits for the next token probabilities
        :param indices: LongTensor of encoded tokens of size (batch_size, length)
        :param lengths: LongTensor of lengths of size (batch_size, )
        :return: FloatTensor of logits of shape (batch_size, length, vocab_size)
        """
        """
        YOUR CODE HERE (⊃｡•́‿•̀｡)⊃━✿✿✿✿✿✿
        Convert indices to embeddings, pass them through recurrent layers
        and apply output linear layer to obtain the logits
        """
        batch_size=indices.shape[0]
        max_len_in_batch=lengths.max().item()
        indices_trimmed=indices[:, :max_len_in_batch]
        embeddings=self.embedding(indices_trimmed)
        rnn_output, _ =self.rnn(embeddings)
        logits=self.linear(rnn_output)
        return logits

    @torch.inference_mode()
    def inference(self, prefix: str = '', temp: float = 1.) -> str:
        """
        Generate new text with an optional prefix
        :param prefix: prefix to start generation
        :param temp: sampling temperature
        :return: generated text
        """
        self.eval()
        device = next(self.parameters()).device
        """
        YOUR CODE HERE (⊃｡•́‿•̀｡)⊃━✿✿✿✿✿✿
        Encode the prefix (do not forget the BOS token!),
        pass it through the model to accumulate RNN hidden state and
        generate new tokens sequentially, sampling from categorical distribution,
        until EOS token or reaching self.max_length.
        Do not forget to divide predicted logits by temperature before sampling
        """
        if prefix.strip():
            try:
                prefix_ids=self.dataset.text2ids(prefix)
            except:
                prefix_ids =[]
        else:
            prefix_ids=[]
        generated_ids=[self.dataset.bos_id] + prefix_ids
        if self.rnn_type ==nn.LSTM:
            hidden=(
                torch.zeros(self.rnn_layers, 1, self.hidden_size, device=device),
                torch.zeros(self.rnn_layers, 1, self.hidden_size, device=device)
            )
        else:
            hidden=torch.zeros(self.rnn_layers, 1, self.hidden_size, device=device)
        if len(generated_ids)>1:
            prefix_tensor=torch.tensor([generated_ids], device=device, dtype=torch.long)
            embeddings=self.embedding(prefix_tensor)
            _, hidden=self.rnn(embeddings, hidden)
        for step in range(self.max_length - len(generated_ids)):
            current_token=torch.tensor([[generated_ids[-1]]], device=device, dtype=torch.long)
            embedding =self.embedding(current_token)
            rnn_out, hidden=self.rnn(embedding, hidden)
            logits=self.linear(rnn_out).squeeze(0).squeeze(0)
            logits=logits / max(temp, 1e-8)
            probs =torch.softmax(logits, dim=-1)
            try:
                next_token =torch.multinomial(probs, 1).item()
            except:
                next_token=torch.argmax(probs).item()
            generated_ids.append(next_token)
            if next_token == self.dataset.eos_id:
                break
        try:
            if generated_ids[-1] == self.dataset.eos_id:
                decode_ids=generated_ids[1:-1]
            else:
                decode_ids=generated_ids[1:]
            generated_text=self.dataset.ids2text(decode_ids)
            if not prefix.strip():
                return generated_text
            else:
                return generated_text
        except Exception as e:
            return prefix if prefix else ""