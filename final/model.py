
import torch


class rPPGModel(torch.nn.Module):
    def __init__(self,input_size=9, d_model=64, nhead=4, ff_hidden_size=128, num_layers=2, output_size=1):
        super(rPPGModel, self).__init__()
        self.input_size = input_size
        self.d_model = d_model
        self.nhead = nhead
        self.ff_hidden_size = ff_hidden_size
        self.num_layers = num_layers
        self.output_size = output_size


        self.encoder_embedding = torch.nn.Linear(self.input_size, self.d_model)
        self.transformer_encoder_layer = torch.nn.TransformerEncoderLayer(d_model=self.d_model, nhead=self.nhead, dim_feedforward=self.ff_hidden_size,batch_first=True,activation="gelu")
        self.transformer_encoder = torch.nn.TransformerEncoder(self.transformer_encoder_layer, num_layers=self.num_layers)

        # self.decoder_embedding = torch.nn.Linear(self.output_size, self.d_model)
        # self.transformer_decoder_layer = torch.nn.TransformerDecoderLayer(d_model=self.d_model, nhead=self.nhead, dim_feedforward=self.ff_hidden_size,batch_first=True,activation="gelu")
        # self.transformer_decoder = torch.nn.TransformerDecoder(self.transformer_decoder_layer, num_layers=self.num_layers)


        self.head = torch.nn.Linear(self.d_model, self.output_size)

    def forward(self, src):
        #predict direction from encoder output 

        #src: (B, T, input_size) , target: (B, T, output_size)
        
        src = self.encoder_embedding(src) #(B, T, d_model)
        memory = self.transformer_encoder(src) #(B, T, d_model)
        output = self.head(memory) #(B, T, output_size)

        return output

        