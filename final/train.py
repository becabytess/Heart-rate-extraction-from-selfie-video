
import os
from torch.utils.data import DataLoader 
from torch.utils.data import random_split 
import torch
import modal 
import sys
sys.path.append("/data") # Tell Python to check your mounted volume directory for modules
from datasets import UBFC_Dataset
# from model import rPPGModel

#modal volume put rppg-data "../data" "data" 
#modal volume put rppg-data datasets.py  dataset.py (add --force flag when uploading again after edits)

image = modal.Image.debian_slim().pip_install("torch", "numpy", "matplotlib","scipy")
app = modal.App("rppg",image=image )

vol = modal.Volume.from_name("rppg-data",create_if_missing=True)

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

        

class MSE_NegPearsonLoss(torch.nn.Module):
    def __init__(self):
        super(MSE_NegPearsonLoss, self).__init__()
        self.mse_loss = torch.nn.MSELoss()
    def forward(self,preds, targs):

        preds = preds.flatten()
        targs = targs.flatten()
        preds_mean = torch.mean(preds)
        targs_mean = torch.mean(targs)
#(B,T,1)
        preds_std = torch.std(preds)
        targs_std = torch.std(targs)
        
        z_preds = (preds - preds_mean )/preds_std 
        z_targs = (targs - targs_mean) / targs_std 

        pearson_corr = torch.mean(z_preds * z_targs)

        mse_loss = self.mse_loss(preds,targs) 

        return  mse_loss , 1-pearson_corr
    



@app.function(
    gpu="T4",
    memory=8192,
    cpu=2,
    timeout=86400,
    volumes={"/data":vol})
def train():
 #extract data if isn't extracted 
    if not os.path.exists("/data/UBFC-RPPG-Dataset"):
        import tarfile
        with tarfile.open("/data/datatar.tar.gz", "r:gz") as tar:
            os.makedirs("/data/data", exist_ok=True)
            tar.extractall(path="data")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_path = os.path.join('data/data','UBFC-RPPG-Dataset')
    subjects = os.listdir(data_path)  
    dataset = UBFC_Dataset(data_path, subjects)

   

    model = rPPGModel(input_size=9, d_model=64, nhead=4, ff_hidden_size=128, num_layers=2, output_size=1)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    random_seed = 42
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size], generator=torch.Generator().manual_seed(random_seed))

    batch_size = 128
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)



    def evaluate():
        model.eval()
        total_loss = 0 
        for color_seq, signal_seq in test_loader:
                color_seq = color_seq.to(device)
                signal_seq = signal_seq.to(device)
                preds = model(color_seq)
                mse_loss,neg_pearson_loss = MSE_NegPearsonLoss()(preds, signal_seq)
                curr_loss = mse_loss + neg_pearson_loss
                total_loss += curr_loss.item()
        

        
        avg_loss =  total_loss / len(test_loader)
        model.train()
        return avg_loss


                
    epochs = 10 
    log_every = 10
    best_loss = float('inf')
    load_weight = True  
    if os.path.exists("/data/best_model.pth") and load_weight:
        print("Loading checkpoint")
        chkpt = torch.load("/data/best_model.pth")
        model.load_state_dict(chkpt["model_state_dict"])
        optimizer.load_state_dict(chkpt["optimizer_state_dict"])
        best_loss = chkpt["loss"]
        
    params = sum(p.numel() for p in model.parameters())
    print(f"....Starting training....Num Parameters: {params}")
    for epoch in range(epochs):
        model.train()
        for idx, (color_seq, signal_seq) in enumerate(train_loader):
            optimizer.zero_grad()
            color_seq = color_seq.to(device)
            signal_seq = signal_seq.to(device)
            preds = model(color_seq)
            mse_loss,neg_pearson_loss = MSE_NegPearsonLoss()(preds, signal_seq)
            loss = mse_loss + neg_pearson_loss
            loss.backward()
            optimizer.step()

            if (idx + 1) % log_every == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Step [{idx+1}/{len(train_loader)}], Mse loss: {mse_loss.item():.4f}, Neg Pearson loss: {neg_pearson_loss.item():.4f}, Total loss: {loss.item():.4f}")
        val_loss = evaluate()
        chkpt = {
            "epoch":epoch,
            "loss":loss.item(),
            "optimizer_state_dict":optimizer.state_dict(),
            "model_state_dict":model.state_dict()

        }
        if val_loss < best_loss:
            print(f"New Best Loss: {val_loss}")
            torch.save(chkpt, "/data/best_model.pth")
        else:
            print(f"No improvement in validation loss: {val_loss} (Best: {best_loss})")
            torch.save(chkpt, "/data/latest.pth")    


@app.local_entrypoint()
def main():
    train.remote()


