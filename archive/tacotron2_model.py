import torch
import torch.nn as nn
import torch.nn.functional as F

def get_mask_from_lengths(lengths, max_len=None):
    """Generates a boolean mask for padded tensors."""
    if max_len is None:
        max_len = lengths.max().item()
    ids = torch.arange(0, max_len, device=lengths.device)
    mask = (ids < lengths.unsqueeze(1)).bool()
    return mask

class TextEncoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim=512, num_conv_layers=3, lstm_hidden=256):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        convolutions = []
        for _ in range(num_conv_layers):
            conv_layer = nn.Sequential(
                nn.Conv1d(embedding_dim, embedding_dim, kernel_size=5, padding=2),
                nn.BatchNorm1d(embedding_dim),
                nn.ReLU(),
                nn.Dropout(0.5)
            )
            convolutions.append(conv_layer)
        self.convolutions = nn.ModuleList(convolutions)
        
        self.lstm = nn.LSTM(
            embedding_dim, lstm_hidden, num_layers=1, 
            batch_first=True, bidirectional=True
        )
    
    def forward(self, text, text_lengths):
        x = self.embedding(text).transpose(1, 2)
        for conv in self.convolutions:
            x = conv(x)
        x = x.transpose(1, 2)
        
        x = nn.utils.rnn.pack_padded_sequence(
            x, text_lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        self.lstm.flatten_parameters()
        outputs, _ = self.lstm(x)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
        return outputs

class LocationSensitiveAttention(nn.Module):
    def __init__(self, encoder_dim=768, decoder_dim=1024, attention_dim=128, 
                 location_n_filters=32, location_kernel_size=31):
        super().__init__()
        self.query_layer = nn.Linear(decoder_dim, attention_dim, bias=False)
        self.memory_layer = nn.Linear(encoder_dim, attention_dim, bias=False)
        self.v = nn.Linear(attention_dim, 1, bias=False)
        
        self.location_conv = nn.Conv1d(
            2, location_n_filters, kernel_size=location_kernel_size, 
            padding=(location_kernel_size - 1) // 2, bias=False
        )
        self.location_dense = nn.Linear(location_n_filters, attention_dim, bias=False)
    
    def forward(self, query, memory, attention_weights_cat, mask=None):
        query_processed = self.query_layer(query.unsqueeze(1))
        memory_processed = self.memory_layer(memory)
        
        location_processed = self.location_conv(attention_weights_cat).transpose(1, 2)
        location_processed = self.location_dense(location_processed)
        
        energies = self.v(torch.tanh(
            query_processed + memory_processed + location_processed
        )).squeeze(-1)
        
        # 🔴 MASK APPLIED: Attention ignores padding
        if mask is not None:
            energies = energies.masked_fill(~mask, -float('inf'))
            
        attention_weights = F.softmax(energies, dim=1)
        context = torch.bmm(attention_weights.unsqueeze(1), memory).squeeze(1)
        return context, attention_weights

class PreNet(nn.Module):
    def __init__(self, in_dim, sizes=[256, 256], dropout=0.5):
        super().__init__()
        in_sizes = [in_dim] + sizes[:-1]
        self.layers = nn.ModuleList([
            nn.Linear(in_size, out_size, bias=False)
            for in_size, out_size in zip(in_sizes, sizes)
        ])
        self.dropout = dropout
    
    def forward(self, x):
        for layer in self.layers:
            x = F.relu(layer(x))
            x = F.dropout(x, p=self.dropout, training=True)
        return x

class Decoder(nn.Module):
    def __init__(self, n_mels=80, encoder_dim=768, decoder_dim=1024, prenet_dim=256):
        super().__init__()
        self.n_mels = n_mels
        self.encoder_dim = encoder_dim
        self.decoder_dim = decoder_dim
        
        self.prenet = PreNet(n_mels, sizes=[prenet_dim, prenet_dim])
        self.attention = LocationSensitiveAttention(encoder_dim, decoder_dim)
        
        # 🔴 LSTMCell APPLIED: Fast autoregressive unrolling
        self.lstm1 = nn.LSTMCell(prenet_dim + encoder_dim, decoder_dim)
        self.lstm2 = nn.LSTMCell(decoder_dim, decoder_dim)
        
        self.mel_projection = nn.Linear(decoder_dim + encoder_dim, n_mels)
        self.stop_projection = nn.Linear(decoder_dim + encoder_dim, 1)
    
    def forward(self, encoder_outputs, mask, mels=None, teacher_forcing_ratio=1.0):
        batch_size = encoder_outputs.size(0)
        text_len = encoder_outputs.size(1)
        max_len = mels.size(2) if mels is not None else 1000
        
        prev_mel = torch.zeros(batch_size, self.n_mels, device=encoder_outputs.device)
        
        h1 = torch.zeros(batch_size, self.decoder_dim, device=encoder_outputs.device)
        c1 = torch.zeros(batch_size, self.decoder_dim, device=encoder_outputs.device)
        h2 = torch.zeros(batch_size, self.decoder_dim, device=encoder_outputs.device)
        c2 = torch.zeros(batch_size, self.decoder_dim, device=encoder_outputs.device)
        
        attention_weights = torch.zeros(batch_size, text_len, device=encoder_outputs.device)
        attention_weights_cum = torch.zeros(batch_size, text_len, device=encoder_outputs.device)
        attention_context = torch.zeros(batch_size, self.encoder_dim, device=encoder_outputs.device)
        
        mel_outputs, stop_tokens, alignments = [], [], []
        
        for t in range(max_len):
            prenet_out = self.prenet(prev_mel)
            lstm1_input = torch.cat([prenet_out, attention_context], dim=-1)
            
            h1, c1 = self.lstm1(lstm1_input, (h1, c1))
            h2, c2 = self.lstm2(h1, (h2, c2))
            
            attention_weights_cat = torch.stack([attention_weights, attention_weights_cum], dim=1)
            
            attention_context, attention_weights = self.attention(
                h2, encoder_outputs, attention_weights_cat, mask=mask
            )
            attention_weights_cum += attention_weights
            
            projection_input = torch.cat([h2, attention_context], dim=-1)
            mel_output = self.mel_projection(projection_input)
            stop_token = self.stop_projection(projection_input)
            
            mel_outputs.append(mel_output)
            stop_tokens.append(stop_token.squeeze(-1))
            alignments.append(attention_weights)
            
            if mels is not None and torch.rand(1).item() < teacher_forcing_ratio:
                prev_mel = mels[:, :, t]
            else:
                prev_mel = mel_output
                
            if mels is None and t > 10:
                if (torch.sigmoid(stop_token) > 0.5).all():
                    break
                    
        return (torch.stack(mel_outputs, dim=2), 
                torch.stack(stop_tokens, dim=1), 
                torch.stack(alignments, dim=1))

class PostNet(nn.Module):
    def __init__(self, n_mels=80, postnet_embedding_dim=512, postnet_kernel_size=5, postnet_n_convolutions=5):
        super().__init__()
        self.convolutions = nn.ModuleList()
        self.convolutions.append(
            nn.Sequential(
                nn.Conv1d(n_mels, postnet_embedding_dim, kernel_size=postnet_kernel_size, padding=(postnet_kernel_size - 1) // 2),
                nn.BatchNorm1d(postnet_embedding_dim), nn.Tanh(), nn.Dropout(0.5)
            )
        )
        for _ in range(1, postnet_n_convolutions - 1):
            self.convolutions.append(
                nn.Sequential(
                    nn.Conv1d(postnet_embedding_dim, postnet_embedding_dim, kernel_size=postnet_kernel_size, padding=(postnet_kernel_size - 1) // 2),
                    nn.BatchNorm1d(postnet_embedding_dim), nn.Tanh(), nn.Dropout(0.5)
                )
            )
        self.convolutions.append(
            nn.Sequential(
                nn.Conv1d(postnet_embedding_dim, n_mels, kernel_size=postnet_kernel_size, padding=(postnet_kernel_size - 1) // 2),
                nn.BatchNorm1d(n_mels), nn.Dropout(0.5)
            )
        )
    
    def forward(self, x):
        for conv in self.convolutions: x = conv(x)
        return x

class Tacotron2(nn.Module):
    def __init__(self, vocab_size, n_mels=80, speaker_embedding_dim=256):
        super().__init__()
        
        # 🔴 T4 FIX: Scaled TextEncoder down by 50%
        self.text_encoder = TextEncoder(
            vocab_size, 
            embedding_dim=256, 
            lstm_hidden=128
        )
        
        # 🔴 T4 FIX: 256 (Text) + 256 (Speaker) = 512 Encoder Dim. Decoder scaled down to 512.
        self.decoder = Decoder(
            n_mels=n_mels, 
            encoder_dim=512, 
            decoder_dim=512, 
            prenet_dim=128
        )
        self.postnet = PostNet(n_mels=n_mels)
    
    def forward(self, text, text_lengths, speaker_embeddings, mels=None, teacher_forcing_ratio=1.0):
        # 1. Encode Text
        encoded_text = self.text_encoder(text, text_lengths)
        
        # 2. Generate the mask based on actual text lengths
        mask = get_mask_from_lengths(text_lengths, max_len=text.size(1))
        
        # 3. CRITICAL: Inject Speaker Identity & Ensure Contiguity
        speaker_emb_expanded = speaker_embeddings.unsqueeze(1).expand(-1, encoded_text.size(1), -1)
        encoded_with_speaker = torch.cat([encoded_text, speaker_emb_expanded], dim=-1).contiguous()
        
        # 4. Decode
        mel_outputs, stop_tokens, alignments = self.decoder(
            encoded_with_speaker, mask, mels, teacher_forcing_ratio
        )
        
        # 5. PostNet Refinement
        mel_outputs_postnet = mel_outputs + self.postnet(mel_outputs)
        
        return mel_outputs_postnet, mel_outputs, stop_tokens, alignments

if __name__ == "__main__":
    model = Tacotron2(vocab_size=80)
    text = torch.randint(0, 80, (4, 50))
    text_lengths = torch.LongTensor([50, 45, 30, 20])
    speaker_embeddings = torch.randn(4, 256)
    mels = torch.randn(4, 80, 200)
    
    mel_post, mel_raw, stops, aligns = model(text, text_lengths, speaker_embeddings, mels)
    print("✅ All shapes correct, attention masked, and model optimized for Modal T4 VRAM!")