import torch
import torch.nn as nn
import matplotlib.pyplot as plt

def trainer(model, train_dataloader, val_dataloader, num_epochs, path_to_save='/home/atharva',
          checkpoint_path='/home/atharva',
          checkpoint=100, train_batch=1, test_batch=1, device='cpu'): # 2 Marks. 
    """
    Everything by default gets shifted to the GPU. Select the device according to your system configuration
    If you do no have a GPU, change the device parameter to "device='cpu'"
    :param model: the Classification model..
    :param train_dataloader: train_dataloader
    :param val_dataloader: val_Dataloader
    :param num_epochs: num_epochs
    :param path_to_save: path to save model
    :param checkpoint_path: checkpointing path
    :param checkpoint: when to checkpoint
    :param train_batch: 1
    :param test_batch: 1
    :param device: Defaults on GPU, pass 'cpu' as parameter to run on CPU. 
    :return: None
    """
    #torch.backends.cudnn.benchmark = True #Comment this if you are not using a GPU...
    # set the network to training mode.
    #model.cuda()  # if gpu available otherwise comment this line. 
    # your code goes here.
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    max_acc = None
    training_loss = []
    training_acc = []
    val_loss = []
    val_acc = []

    for epoch in range(1, num_epochs+1):
        if epoch%checkpoint==0:
            torch.save({
            'epoch':epoch,
            'optimizer': optimizer.state_dict(),
            'model': model.state_dict(),
            'train_loss': training_loss,
            'val_loss': val_loss,
            'train_acc' : training_acc,
            'val_acc' : val_acc
            }, checkpoint_path+'/checkpoint.pt')

            exit(10)

        model.train()
        epoch_loss_train = 0
        epoch_acc_train = 0

        for _,data in enumerate(train_dataloader):
            optimizer.zero_grad()

            sentence = data['statement'].to(device)
            justification = data['justification'].to(device)
            credit_history = data['credit_history'].to(device)
            label = data['label'].to(device)

            output = model(sentence, justification, credit_history)

            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            epoch_loss_train+=loss.item()
            __, predicted = torch.max(output.data, 1)
            epoch_acc_train += (predicted == label).sum().item()
            del sentence, justification, credit_history, label

        training_loss.append(epoch_loss_train / (_*train_batch)) # loss per batch
        training_acc.append(epoch_acc_train/(_*train_batch)) # acc per batch

        with torch.no_grad():  # No propagating losses here
            model.eval()  
            epoch_loss_val = 0
            epoch_acc_val = 0

            for _, data in enumerate(val_dataloader):

                sentence = data['statement'].to(device)
                justification = data['justification'].to(device)
                credit_history = data['credit_history'].to(device)
                label = data['label'].to(device)

                output = model(sentence, justification, credit_history)

                loss = criterion(output, label)
                epoch_loss_val += loss.item()
                __, predicted = torch.max(output.data, 1)
                epoch_acc_val += (predicted == label).sum().item()

        val_loss.append(epoch_loss_val/(_*test_batch))
        val_acc.append(epoch_acc_val/(_*test_batch))

        if max_acc is None:
              max_acc = epoch_acc_val/(_*test_batch)
        else:
              if (epoch_acc_val/(_*test_batch)) > max_acc:
                    print('saving at validation acc= ', epoch_acc_val/(_*test_batch))
                    torch.save(model.state_dict(), path_to_save+'/model.pth')
                    max_acc = epoch_acc_val/(_*test_batch)

        plt.plot(training_acc)
        plt.plot(val_acc)
        plt.plot(training_loss)
        plt.plot(val_loss)
        plt.show()

