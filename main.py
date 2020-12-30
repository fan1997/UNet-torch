from model.unet import UNet
from model.loss import CrossEntropyWithLogits
from data.dataloader import create_dataset
import torch
from torch.optim import Adam
from tqdm import tqdm
import time

num_epoches = 400
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)


data_dir = "/userhome/Unet/unet/data/"
train_dataloader, val_dataloader = create_dataset(
    data_dir, repeat=1, train_batch_size=12, augment=True)
model = UNet(1, 2).to(device)
criterion = CrossEntropyWithLogits().to(device)
optimizer = Adam(model.parameters(), lr=0.0001, weight_decay=0.0005)

save_step = 400

# TODO: Initialization the params
val_loss = -1
step_now = 0
total_loss = 0
torch.cuda.synchronize()
start = time.time()

for i in range(num_epoches):
    with tqdm(train_dataloader) as tl:
        for (img, mask) in tl:
            step_now += 1
            optimizer.zero_grad()
            pred = model(img.to(device))
            loss = criterion(pred, mask.to(device))
            loss.backward()
            optimizer.step()
            loss_cpu = loss.cpu().item()
            total_loss += loss_cpu
            if step_now % save_step == 0:
                val_loss = 0
                torch.save(model.state_dict(), "./result/step=%d" % (step_now))
                for (img_val, mask_val) in val_dataloader:
                    pred = model(img_val.to(device))
                    loss = criterion(pred, mask_val.to(device))
                    val_loss += loss.cpu().item()
            tl.set_postfix(loss=loss_cpu, avg_loss=total_loss /
                       step_now, val_loss=val_loss/len(val_dataloader))

torch.cuda.synchronize()
end = time.time()
print("training time: ", end - start)
torch.save(model.state_dict(), "./result/step=all")
