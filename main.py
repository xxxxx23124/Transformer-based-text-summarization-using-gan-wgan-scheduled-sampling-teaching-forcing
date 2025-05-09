from Trainer_DataLoader import Controller
from Generator import Transformer as GTransformer
from Discriminator_GAN import Transformer as DTransformer
from Discriminator_WGAN import Transformer as WDTransformer
from transformers import AutoTokenizer
if __name__ == "__main__":
    Controller(GTransformer(AutoTokenizer.from_pretrained("google-bert/bert-large-cased").vocab_size), DTransformer(AutoTokenizer.from_pretrained("google-bert/bert-large-cased").vocab_size)).gan_professor_forcing_train()
    #Controller(GTransformer(AutoTokenizer.from_pretrained("google-bert/bert-large-cased").vocab_size), WDTransformer(AutoTokenizer.from_pretrained("google-bert/bert-large-cased").vocab_size)).wgan_professor_forcing_train()
    #Controller(GTransformer(AutoTokenizer.from_pretrained("google-bert/bert-large-cased").vocab_size)).basic_train()