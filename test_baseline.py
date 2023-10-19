from baseline import *
import torch

data = setup_data()
train_dataset, val_dataset, test_dataset = setup_datasets(data)

train_dataloader = setup_dataloader(train_dataset, shuffle=True)
val_dataloader = setup_dataloader(val_dataset, shuffle=False)
test_dataloader = setup_dataloader(test_dataset, shuffle=False, batch_size=1)

model = BERTModule()

epochs = 10
model, tokenizer = train_loop(model=model, epochs=epochs, train_loader=train_dataloader, val_loader=val_dataloader, verbose=True)
torch.save(model, f"models_saved/run_1_{epochs}.pt")

test(model, tokenizer, test_dataloader)
