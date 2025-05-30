from sentencepiece import SentencePieceProcessor
from torch import tensor, ones, int64, concat, no_grad, load
from torch.nn.utils.rnn import pad_sequence

from ELEC825Transformer2 import Transformer

if __name__ == '__main__':
    device = 'cpu'

    en_model = SentencePieceProcessor()
    en_model.Load('dataset/en.model')
    fr_model = SentencePieceProcessor()
    fr_model.Load('dataset/fr.model')

    en = ['How are you today?',
          'Today''s news tells me, a man cannot live without a woman.',
          'It looks like you''ve shared an image!',
          'How can I help with it?',
          'Let me know what you''re thinking!',
          'Process finished with exit code 0',
          'Testing this would be necessary.',
          'If the model works correctly after loading with strict=False, then this is a valid solution.',
          'The user might also want to suppress the warning about weights_only by setting it to True if possible, but that''s a separate issue.'
          ]
    i = pad_sequence([tensor(en_model.Encode(x, add_bos=True, add_eos=True)[:128],
                             device=device) for x in en], batch_first=True)
    o = ones((len(en), 1), dtype=int64, device=device)

    model = Transformer(en_model.vocab_size(), fr_model.vocab_size(), 512, 2048, 8, 8, 256)
    model.load_state_dict(load('checkpoint/model-3.pth'), strict=False)
    model.eval()

    with no_grad():
        while o.size(1) < 128:
            o = concat((o, model(i, o)[:, -1:, :].argmax(dim=2)), dim=1)

    fr = [fr_model.Decode(x[:(x.index(2) + 1 if 2 in x else len(x))]) for x in o.tolist()]
    print('\n'.join(fr))
