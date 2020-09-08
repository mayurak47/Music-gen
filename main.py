from utils.util import *
from datasets.MusicDataset import MusicDataset
from sklearn.model_selection import train_test_split
from torch import optim
from torch.utils.data import DataLoader
from models.CNN import CNN
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
import matplotlib.pyplot as plt

def main():
    songs = get_notes()

    vocab_set = set()
    for song in songs:
        for note in song:
            vocab_set.add(note)

    n_in, n_out = prep_sequences(songs, sequence_length=100)
    X_train, X_val, y_train, y_val = train_test_split(n_in, n_out, test_size=0.2)

    train_ds = MusicDataset(X_train, y_train)
    val_ds = MusicDataset(X_val, y_val)

    train_dataloader = DataLoader(train_ds, batch_size=512, shuffle=True, num_workers=0)
    val_dataloader = DataLoader(val_ds, batch_size=512, shuffle=False, num_workers=0)

    model = CNN(100, len(vocab_set))
    model.cuda()
    epochs = 25
    initial_lr = 0.001
    optimizer = optim.Adam(model.parameters(), lr=initial_lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    loss_fn = CrossEntropyLoss()

    train_losses = []
    val_losses = []

    train_accuracies = []
    val_accuracies = []

    for epoch in tqdm(range(1, epochs+1)):

        model.train()
        train_loss_total = 0.0
        num_steps = 0
        correct = 0
        ### Train
        for i, batch in enumerate(train_dataloader):
            X, y = batch[0].cuda(), batch[1].cuda()
            train_preds = model(X)

            loss = loss_fn(train_preds, y)
            train_loss_total += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            num_steps += 1

            train_preds = torch.max(train_preds, 1)[1]
            correct += (train_preds == y).float().sum()

        
        train_loss_total_avg = train_loss_total / num_steps
        train_accuracy = correct/len(train_ds)
        train_accuracies.append(train_accuracy)
        train_losses.append(train_loss_total_avg)

        model.eval()
        val_loss_total = 0.0
        num_steps = 0
        correct = 0
        for i, batch in enumerate(val_dataloader):
            with torch.no_grad():
                X, y = batch[0].cuda(), batch[1].cuda()

                val_preds = model(X)
                loss = loss_fn(val_preds, y)
                val_loss_total += loss.item()
                val_preds = torch.max(val_preds, 1)[1]
                correct += (val_preds == y).float().sum()

            num_steps += 1

        val_loss_total_avg = val_loss_total / num_steps
        val_accuracy = correct/len(val_ds)
        val_accuracies.append(val_accuracy)
        val_losses.append(val_loss_total_avg)

        scheduler.step()
        print('\nTrain loss: {:.4f}'.format(train_loss_total_avg))
        print('Train accuracy: {:.4f}'.format(train_accuracy))
        # torch.save(model.state_dict(), "/content/drive/My Drive/epochCNN"+str(epoch)) 
        print('Val loss: {:.4f}'.format(val_loss_total_avg))
        print('Val accuracy: {:.4f}'.format(val_accuracy))

        torch.save(model.state_dict(), "weights/model_params_epoch"+str(epoch))
        torch.save(optimizer.state_dict(), "weights/optim_params_epoch"+str(epoch))

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.plot(range(len(train_accuracies)), train_accuracies)
    plt.plot(range(len(val_accuracies)), val_accuracies)
    plt.savefig("plots/accuracies.png")
    plt.close()

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.plot(range(len(train_losses)), train_losses)
    plt.plot(range(len(val_losses)), val_losses)
    plt.savefig("plots/losses.png")
    plt.close()
    
    generate_midi(model, val_ds, vocab_set, output_filename="output.mid"):

if __name__ == "__main__":
    main()
    