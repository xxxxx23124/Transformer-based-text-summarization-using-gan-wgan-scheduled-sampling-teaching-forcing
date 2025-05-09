from torch.utils.data import Dataset
from datasets import load_dataset
import pandas
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer
import os
from tqdm import tqdm
from torch.utils.data import DataLoader
import random
import time

class XSumDataset(Dataset):
    def __init__(self, state, clean:bool=True):
        assert state in ["train", "validation", "test"]
        xsum_dataset = load_dataset("EdinburghNLP/xsum")
        self.state = state
        self.dataset = xsum_dataset[state]
        if clean:
            remove_list = self.clean()
            self.dataset = self.dataset.filter(lambda x:x['id'] not in remove_list)
        self.dataset_len = len(self.dataset)

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        return {
            'document': sample['document'],
            'summary': sample['summary']
        }

    def info(self):
        document_len_list = pandas.Series([len(x['document']) for x in self.dataset])
        summary_len_list = pandas.Series([len(x['summary']) for x in self.dataset])
        print(self.state+" document describe:")
        print("Elements",len(document_len_list))
        print(document_len_list.describe())
        print(self.state+" summary describe:")
        print("Elements", len(summary_len_list))
        print(summary_len_list.describe())

    def clean(self):
        excluded_ids_list = []
        for x in self.dataset:
            if len(x['document']) == 0 or len(x['summary']) == 0:
                excluded_ids_list.append(x['id'])
            elif len(x['document'])>1000:
                excluded_ids_list.append(x['id'])
        return excluded_ids_list


from dataclasses import dataclass
@dataclass
class Config():
    accum_steps = 1
    epochs = 100000
    need_load_model = False
    need_save_model = False
    save_model_every_steps = 2000
    generator_path = "./model_checkpoints/generator.pt"
    discriminator_path = ""
    generator_lr=1e-4
    discriminator_lr = 1e-4
    use_professor_forcing = True
    train_batch_size = 2
    val_batch_size = 1
    test_batch_size = 1
    tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-large-cased")
    vocab_size = tokenizer.vocab_size
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sampling_prob = 0

class Tools(Config):
    def collate_fn(self, batch):
        documents = [item['document'] for item in batch]
        summaries = [item['summary'] for item in batch]
        # Tokenize documents
        document_encodings = self.tokenizer(
            documents,
            padding=True,
            truncation=False,
            return_tensors="pt"
        )
        # Tokenize summaries
        summary_encodings = self.tokenizer(
            summaries,
            padding=True,
            truncation=False,
            return_tensors="pt"
        )
        tgt_len = summary_encodings['input_ids'].shape[1]
        causal_mask = torch.tril(torch.ones((tgt_len, tgt_len))).int()  # [tgt_len, tgt_len]
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, tgt_len, tgt_len]
        key_mask = summary_encodings['attention_mask'].unsqueeze(1).unsqueeze(2)  # [batch_size, 1, 1, tgt_len]
        tgt_mask = causal_mask & key_mask  # [batch_size, 1, tgt_len, tgt_len]
        return {
            'document_input_ids': document_encodings['input_ids'],
            'document_attention_mask': document_encodings['attention_mask'],
            'summary_input_ids': summary_encodings['input_ids'],
            'summary_attention_mask': tgt_mask,
            'summary_attention_mask_original':summary_encodings['attention_mask']
        }

    def __init__(self, generator, discriminator):
        super().__init__()
        self.generator = generator.to(self.device)
        self.generator_optimizer = torch.optim.RAdam(
                            self.generator.parameters(),
                            lr=self.generator_lr,
                        )
        if self.use_professor_forcing:
            assert discriminator is not None
            self.discriminator = discriminator.to(self.device)
            self.discriminator_optimizer = torch.optim.RAdam(
                self.discriminator.parameters(),
                lr=self.discriminator_lr,
            )

        self.crossEntropyLoss = nn.CrossEntropyLoss()
        self.bceLoss = torch.nn.BCELoss()
        self.train_loader = DataLoader(
            XSumDataset("train", clean=True),
            batch_size=self.train_batch_size,
            shuffle=True,
            drop_last=True,
            collate_fn=lambda batch: self.collate_fn(batch)
        )
        self.val_loader = DataLoader(
            XSumDataset("validation", clean=True),
            batch_size=self.val_batch_size,
            shuffle=True,
            drop_last=True,
            collate_fn=lambda batch: self.collate_fn(batch)
        )
        self.test_loader = DataLoader(
            XSumDataset("test", clean=True),
            batch_size=self.test_batch_size,
            shuffle=True,
            drop_last=True,
            collate_fn=lambda batch: self.collate_fn(batch)
        )

    def load_model(self, filepath):
        try:
            checkpoint = torch.load(filepath)
            print("找到模型: " + filepath)
            return checkpoint
        except FileNotFoundError:
            print("没有找到模型: " + filepath)

    def save_model(self, filepath, model, optimizer):
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'tokenizer': self.tokenizer
        }, filepath)
        print(f"模型已保存")

    def teaching_forcing_step(self, batch):
        src = batch['document_input_ids'].to(self.device)
        src_mask = batch['document_attention_mask'].to(self.device)
        tgt = batch['summary_input_ids'].to(self.device)
        tgt_mask = batch['summary_attention_mask'].to(self.device)
        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]
        tgt_mask_input = tgt_mask[:, :, :tgt_input.shape[1], :tgt_input.shape[1]]
        return self.crossEntropyLoss(
            self.generator(src, tgt_input, src_mask, tgt_mask_input, src_mask).view(-1, self.vocab_size),
            tgt_output.contiguous().view(-1))

    def scheduled_sampling_step(self, batch):
        src = batch['document_input_ids'].to(self.device)
        src_mask = batch['document_attention_mask'].to(self.device)
        tgt = batch['summary_input_ids'].to(self.device)
        tgt_mask = batch['summary_attention_mask'].to(self.device)
        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]
        tgt_mask_input = tgt_mask[:, :, :tgt_input.shape[1], :tgt_input.shape[1]]
        memory = self.generator.encoder(src, src_mask)
        logits = self.generator.decoder(tgt_input, memory, tgt_mask=tgt_mask_input, memory_mask=src_mask)
        probs = torch.softmax(logits, dim=-1)
        sampled_ids = torch.multinomial(probs, num_samples=1)
        batch_size = tgt_input.size()[0]
        tgt_len = tgt_input.size()[1]
        sampling_mask = (torch.rand(batch_size, tgt_len - 1, device=self.device) < self.sampling_prob).long()
        tgt_input[:, 1:] = tgt_input[:, 1:] * (1 - sampling_mask) + sampled_ids[:, :-1] * sampling_mask
        return self.crossEntropyLoss(
            self.generator.decoder(tgt_input, memory, tgt_mask=tgt_mask_input, memory_mask=src_mask).view(-1,self.vocab_size),
            tgt_output.contiguous().view(-1))

    def generate_summary_logits(self, document_input_ids, document_attention_mask, target_length):
        batch_size = document_input_ids.size(0)
        memory = self.generator.encoder(document_input_ids, document_attention_mask)
        tgt_input = torch.full((batch_size, 1), self.tokenizer.cls_token_id, device=self.device)
        generated_logits = torch.zeros(batch_size, 1, self.vocab_size, device=self.device, dtype=torch.float)
        generated_logits[:, :, self.tokenizer.cls_token_id] = 1.0
        generated_logits = generated_logits + torch.randn(batch_size, 1, self.vocab_size, device=self.device) * 0.1
        target_length = target_length - 1
        for _ in range(target_length):
            tgt_mask = torch.tril(torch.ones((batch_size, 1, tgt_input.size(1), tgt_input.size(1)), device=self.device)).int()
            outputs = self.generator.decoder(tgt_input, memory, tgt_mask=tgt_mask, memory_mask=document_attention_mask)
            logits = outputs[:, -1, :]
            generated_logits = torch.cat([generated_logits, logits.unsqueeze(1)], dim=1)
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            tgt_input = torch.cat([tgt_input, next_token], dim=1)
        return generated_logits

    def gan_generator_step(self, src, src_mask, tgt_mask, tgt_len):
        batchsize = src.size(0)
        valid = torch.full((batchsize, 1), 0.90, dtype=torch.float32)
        valid_ = valid + 0.1 * torch.rand(valid.shape, dtype=torch.float32)
        valid_ = valid_.to(self.device)
        generate_summary_logits = self.generate_summary_logits(src, src_mask, tgt_len)
        g_loss = self.bceLoss(self.discriminator(src, generate_summary_logits, src_mask, tgt_mask, src_mask).mean(dim=1), valid_)
        return g_loss

    def gan_discriminator_step(self, src, src_mask, tgt_summary_logits, tgt_mask, tgt_len):
        batchsize = src.size(0)
        valid = torch.full((batchsize, 1), 0.9, dtype=torch.float32)
        valid_ = valid + 0.1 * torch.rand(valid.shape, dtype=torch.float32)
        valid_ = valid_.to(self.device)
        fake = torch.full((batchsize, 1), 0., dtype=torch.float32)
        fake_ = fake + 0.1 * torch.rand(fake.shape, dtype=torch.float32)
        fake_ = fake_.to(self.device)
        with torch.no_grad():
            generate_summary_logits = self.generate_summary_logits(src, src_mask, tgt_len)
        generate_summary_logits.requires_grad_()
        real_loss = self.bceLoss(self.discriminator(src, tgt_summary_logits, src_mask, tgt_mask, src_mask).mean(dim=1), valid_)
        fake_loss = self.bceLoss(self.discriminator(src, generate_summary_logits, src_mask, tgt_mask, src_mask).mean(dim=1), fake_)
        return real_loss, fake_loss

    def gan_professor_forcing_step(self, batch):
        src = batch['document_input_ids'].to(self.device)
        src_mask = batch['document_attention_mask'].to(self.device)
        tgt = batch['summary_input_ids']
        tgt_mask = batch['summary_attention_mask_original'].to(self.device)
        batch_size, tgt_len = tgt.shape
        tgt_one_hot = F.one_hot(tgt, num_classes=self.vocab_size)
        tgt_summary_logits = tgt_one_hot + torch.randn(batch_size, tgt_len, self.vocab_size) * 0.1
        tgt_summary_logits = tgt_summary_logits.to(self.device)
        g_loss = self.gan_generator_step(src, src_mask, tgt_mask, tgt_len)
        real_loss, fake_loss = self.gan_discriminator_step(src, src_mask, tgt_summary_logits, tgt_mask, tgt_len)
        return g_loss, real_loss, fake_loss

    def wgan_generator_step(self, src, src_mask, tgt_mask, tgt_len):
        generate_summary_logits = self.generate_summary_logits(src, src_mask, tgt_len)
        g_loss = -torch.mean(self.discriminator(src, generate_summary_logits, src_mask, tgt_mask, src_mask).mean(dim=1))
        return g_loss

    def compute_grad2(self, d_out, x_in):
        batch_size = x_in.size(0)
        grad_dout = torch.autograd.grad(
            outputs=d_out.sum(), inputs=x_in,
            create_graph=True, retain_graph=True, only_inputs=True
            )[0]
        grad_dout2 = grad_dout.pow(2)
        assert (grad_dout2.size() == x_in.size())
        reg = grad_dout2.view(batch_size, -1).sum(1)
        return reg

    def gradient_penalty(self, src, src_mask, tgt_mask, tgt_summary_logits, generate_summary_logits, center=1.):
        batch_size = tgt_summary_logits.size(0)
        eps = torch.rand(batch_size, device=self.device).view(batch_size, 1, 1)
        x_interp = (1 - eps) * tgt_summary_logits + eps * generate_summary_logits
        x_interp = x_interp.detach()
        x_interp.requires_grad_()
        d_out = self.discriminator(src, x_interp, src_mask, tgt_mask, src_mask).mean(dim=1)
        reg = (self.compute_grad2(d_out, x_interp).sqrt() - center).pow(2).mean()
        return reg

    def wgan_discriminator_step(self, src, src_mask, tgt_summary_logits, tgt_mask, tgt_len):
        with torch.no_grad():
            generate_summary_logits = self.generate_summary_logits(src, src_mask, tgt_len)
        generate_summary_logits.requires_grad_()
        pred_r = self.discriminator(src, tgt_summary_logits, src_mask, tgt_mask, src_mask).mean(dim=1)
        real_loss = -torch.mean(pred_r)
        pred_f = self.discriminator(src, generate_summary_logits, src_mask, tgt_mask, src_mask).mean(dim=1)
        fake_loss = torch.mean(pred_f)
        gp = 10 * self.gradient_penalty(src, src_mask, tgt_mask, tgt_summary_logits, generate_summary_logits)
        return real_loss, fake_loss, gp

    def wgan_professor_forcing_step(self, batch):
        src = batch['document_input_ids'].to(self.device)
        src_mask = batch['document_attention_mask'].to(self.device)
        tgt = batch['summary_input_ids']
        tgt_mask = batch['summary_attention_mask_original'].to(self.device)
        batch_size, tgt_len = tgt.shape
        tgt_one_hot = F.one_hot(tgt, num_classes=self.vocab_size)
        tgt_summary_logits = tgt_one_hot + torch.randn(batch_size, tgt_len, self.vocab_size) * 0.1
        tgt_summary_logits = tgt_summary_logits.to(self.device)
        g_loss = self.wgan_generator_step(src, src_mask, tgt_mask, tgt_len)
        real_loss, fake_loss, gp = self.wgan_discriminator_step(src, src_mask, tgt_summary_logits, tgt_mask, tgt_len)
        return g_loss, real_loss, fake_loss, gp


class Controller(Tools):
    def __init__(self, generator, discriminator = None):
        super().__init__(generator=generator, discriminator = discriminator)
        print("generator parameters: ", sum(p.numel() for p in self.generator.parameters()))
        print("discriminator parameters: ", sum(p.numel() for p in self.discriminator.parameters()))

        if self.need_load_model:
            try:
                checkpoint = self.load_model(self.generator_path)
                self.generator.load_state_dict(checkpoint['model_state_dict'])
                self.generator_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                print("模型加载成功")
            except TypeError:
                print("模型加载失败")

    def basic_train(self):
        for epoch in range(self.epochs):
            train_loader_len = len(self.train_loader)
            pbar = tqdm(self.train_loader)
            self.generator_optimizer.zero_grad()
            random.seed(time.time())
            for i, batch in enumerate(pbar):
                self.sampling_prob = i / train_loader_len
                if random.random() < self.sampling_prob:
                    loss = self.scheduled_sampling_step(batch)
                else:
                    loss = self.teaching_forcing_step(batch)
                loss = loss / self.accum_steps
                loss.backward()
                pbar.set_postfix({'loss': loss.item() * self.accum_steps, 'sampling_prob': self.sampling_prob})
                if (i + 1) % self.accum_steps == 0 or (i + 1) == len(self.train_loader):
                    torch.nn.utils.clip_grad_norm_(self.generator.parameters(), 1.0)
                    self.generator_optimizer.step()
                    self.generator_optimizer.zero_grad()
            if self.need_save_model:
                self.save_model(self.generator_path,self.generator,self.generator_optimizer)

    def gan_professor_forcing_train(self):
        self.discriminator_path = "./model_checkpoints/discriminator_gan.pt"
        if self.need_load_model:
            try:
                checkpoint = self.load_model(self.discriminator_path)
                self.discriminator.load_state_dict(checkpoint['model_state_dict'])
                self.discriminator_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                print("模型加载成功")
            except TypeError:
                print("模型加载失败")
        for epoch in range(self.epochs):
            pbar = tqdm(self.train_loader)
            self.generator_optimizer.zero_grad()
            self.discriminator_optimizer.zero_grad()
            for i, batch in enumerate(pbar):
                g_loss, real_loss, fake_loss = self.gan_professor_forcing_step(batch)
                g_loss = g_loss / self.accum_steps
                g_loss.backward()
                real_loss = real_loss / self.accum_steps
                real_loss.backward()
                fake_loss = fake_loss / self.accum_steps
                fake_loss.backward()
                pbar.set_postfix({'g_loss': g_loss.item() * self.accum_steps,
                                  'real_loss': real_loss.item() * self.accum_steps, 'fake_loss': fake_loss.item() * self.accum_steps})
                if (i + 1) % self.accum_steps == 0 or (i + 1) == len(self.train_loader):
                    torch.nn.utils.clip_grad_norm_(self.generator.parameters(), 1.0)
                    torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), 1.0)
                    self.generator_optimizer.step()
                    self.generator_optimizer.zero_grad()
                    self.discriminator_optimizer.step()
                    self.discriminator_optimizer.zero_grad()
                if self.need_save_model and ((i + 1) % self.save_model_every_steps == 0 or (i + 1) == len(self.train_loader)):
                    self.save_model(self.generator_path,self.generator,self.generator_optimizer)
                    self.save_model(self.discriminator_path, self.discriminator, self.discriminator_optimizer)

    def wgan_professor_forcing_train(self):
        self.discriminator_path = "./model_checkpoints/discriminator_wgan.pt"
        if self.need_load_model:
            try:
                checkpoint = self.load_model(self.discriminator_path)
                self.discriminator.load_state_dict(checkpoint['model_state_dict'])
                self.discriminator_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                print("模型加载成功")
            except TypeError:
                print("模型加载失败")
        for epoch in range(self.epochs):
            pbar = tqdm(self.train_loader)
            self.generator_optimizer.zero_grad()
            self.discriminator_optimizer.zero_grad()
            for i, batch in enumerate(pbar):
                g_loss, real_loss, fake_loss, gp = self.wgan_professor_forcing_step(batch)
                g_loss = g_loss / self.accum_steps
                g_loss.backward()
                real_loss = real_loss / self.accum_steps
                real_loss.backward()
                fake_loss = fake_loss / self.accum_steps
                fake_loss.backward()
                gp = gp / self.accum_steps
                gp.backward()
                pbar.set_postfix({'g_loss': g_loss.item() * self.accum_steps,
                                  'real_loss': real_loss.item() * self.accum_steps, 'fake_loss': fake_loss.item() * self.accum_steps, 'gp': gp.item() * self.accum_steps})
                if (i + 1) % self.accum_steps == 0 or (i + 1) == len(self.train_loader):
                    torch.nn.utils.clip_grad_norm_(self.generator.parameters(), 1.0)
                    torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), 1.0)
                    self.generator_optimizer.step()
                    self.generator_optimizer.zero_grad()
                    self.discriminator_optimizer.step()
                    self.discriminator_optimizer.zero_grad()
                if self.need_save_model and ((i + 1) % self.save_model_every_steps == 0 or (i + 1) == len(self.train_loader)):
                    self.save_model(self.generator_path,self.generator,self.generator_optimizer)
                    self.save_model(self.discriminator_path, self.discriminator, self.discriminator_optimizer)



