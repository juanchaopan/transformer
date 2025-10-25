from torch import device, tensor, no_grad, save
from torch.cuda import is_available
from torch.nn import CrossEntropyLoss
from torch.nn.utils.rnn import pad_sequence
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from ELEC825Dataset import WikimediaDataset, HuggingFaceDataset
from ELEC825Transformer import Transformer


def collate(batch):
    i = pad_sequence([tensor([1] + x[0][:max_length - 2] + [2]) for x in batch], batch_first=True)
    o = pad_sequence([tensor([1] + x[1][:max_length - 1]) for x in batch], batch_first=True)
    t = pad_sequence([tensor(x[1][:max_length - 1] + [2]) for x in batch], batch_first=True)
    return i, o, t


def train():
    loss_fn = CrossEntropyLoss()
    optimizer = Adam(model.parameters(), learning_rate)
    writer = SummaryWriter(f'{log}-{experiment}')

    global_index = 0
    for epoch in range(num_epochs):
        model.train()
        for index, (i, o, t) in enumerate(train_loader):
            i = i.to(dev)
            o = o.to(dev)
            t = t.to(dev)

            p = model.forward(i, o)
            mask = t != 0
            logit = p[mask]
            target = t[mask]
            loss = loss_fn(logit, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            writer.add_scalar('train-loss', loss.item(), global_index)
            global_index += 1

        model.eval()
        with no_grad():
            total_loss = 0
            for index, (i, o, t) in enumerate(validation_loader):
                i = i.to(dev)
                o = o.to(dev)
                t = t.to(dev)

                p = model.forward(i, o)
                mask = t != 0
                logit = p[mask]
                target = t[mask]
                loss = loss_fn(logit, target)
                total_loss += loss.item()
            avg_loss = total_loss / len(validation_loader)
            writer.add_scalar('validation-epoch-loss', avg_loss, epoch)

        save(model.state_dict(), f'{checkpoint}/model-{epoch}.pth')

    model.eval()
    with no_grad():
        total_loss = 0
        for index, (i, o, t) in enumerate(test_loader):
            i = i.to(dev)
            o = o.to(dev)
            t = t.to(dev)

            p = model.forward(i, o)
            mask = t != 0
            logit = p[mask]
            target = t[mask]
            loss = loss_fn(logit, target)
            total_loss += loss.item()
        avg_loss = total_loss / len(test_loader)
        writer.add_scalar('test-loss', avg_loss, 0)

    writer.close()


if __name__ == '__main__':
    checkpoint = 'checkpoint'
    log = 'log/experiment'
    experiment = 5
    learning_rate = 1e-4
    batch_size = 32
    num_epochs = 30
    dev = device('cuda' if is_available() else 'cpu')
    max_length = 256
    src_size = 50000
    tgt_size = 50000
    d_model = 512
    d_ffn = 2048
    num_heads = 8
    num_layers = 8

    train_dataset = WikimediaDataset(
        'dataset/en.model',
        'dataset/fr.model',
        'dataset/wikimedia.en-fr.en',
        'dataset/wikimedia.en-fr.fr')
    validation_dataset = HuggingFaceDataset(
        'dataset/en.model',
        'dataset/fr.model',
        'dataset/wmt14-validation', 'en', 'fr', )
    test_dataset = HuggingFaceDataset(
        'dataset/en.model',
        'dataset/fr.model',
        'dataset/wmt14-test', 'en', 'fr', )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate, shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, collate_fn=collate, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=collate, shuffle=True)

    model = Transformer(src_size, tgt_size, d_model, d_ffn, num_heads, num_layers, max_length)
    model.to(dev)
    train()
