from sentencepiece import SentencePieceTrainer

if __name__ == '__main__':
    SentencePieceTrainer.Train(
        input='dataset/wikimedia.en-fr.en',
        model_prefix='dataset/en',
        vocab_size=50000,
        character_coverage=1.0,
        model_type='unigram',
        user_defined_symbols='<pad>')

    SentencePieceTrainer.Train(
        input='dataset/wikimedia.en-fr.fr',
        model_prefix='dataset/fr',
        vocab_size=50000,
        character_coverage=1.0,
        model_type='unigram',
        user_defined_symbols='<pad>')
