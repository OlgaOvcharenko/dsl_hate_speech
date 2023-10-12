from baseline import *

data = setup_data()
train_dataset, val_dataset, test_dataset = setup_datasets(data)

train_dataloader = setup_dataloader(train_dataset, shuffle=True)
val_dataloader = setup_dataloader(val_dataset, shuffle=False)
test_dataloader = setup_dataloader(test_dataset, shuffle=False)

model = BERTModule()

train_loop(model=model, epochs=1, train_loader=train_dataloader, val_loader=val_dataloader, verbose = True)
torch.save(model, "models_local/run1.pt")
